import copy
import datetime
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import PemsBay
from tsl.nn.metrics import MaskedMAE, MaskedMetric, MaskedMRE, MaskedMSE
from tsl.nn.utils import casting
from tsl.ops.imputation import add_missing_values
from tsl.utils import numpy_metrics, parser_utils
from tsl.utils.parser_utils import ArgParser
import yaml


def get_model_classes(model_str):
    if model_str == "spin":
        model, filler = SPINModel, SPINImputer
    elif model_str == "spin_h":
        model, filler = SPINHierarchicalModel, SPINImputer
    else:
        raise ValueError(f"Model {model_str} not available.")
    return model, filler


def get_dataset(dataset_name: str):
    # build missing dataset
    if dataset_name.endswith("_point"):
        p_fault, p_noise = 0.0, 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith("_block"):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    if dataset_name == "bay":
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=56789)
    raise ValueError(f"Invalid dataset name: {dataset_name}.")


def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = {"eta_min": 0.1 * args.lr, "T_max": args.epochs}
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="spin_h")
    parser.add_argument("--dataset-name", type=str, default="bay_block")
    parser.add_argument("--config", type=str, default="imputation/spin_h.yaml")

    # Splitting/aggregation params
    parser.add_argument("--val-len", type=float, default=0.1)
    parser.add_argument("--test-len", type=float, default=0.2)

    # Training params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--batches-epoch", type=int, default=300)
    parser.add_argument("--batch-inference", type=int, default=32)
    parser.add_argument("--split-batch-in", type=int, default=1)
    parser.add_argument("--grad-clip-val", type=float, default=5.0)
    parser.add_argument("--loss-fn", type=str, default="l1_loss")
    parser.add_argument("--lr-scheduler", type=str, default=None)

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, imputer_cls = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = imputer_cls.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, "r") as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, imputer_class = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name)

    logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name, args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "config.yaml"), "w") as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # time embedding
    time_emb = dataset.datetime_encoded(["day", "week"]).values
    exog_map = {"global_temporal_encoding": time_emb}
    input_map = {"u": "temporal_encoding", "x": "data"}

    adj = dataset.get_connectivity(threshold=args.adj_threshold, include_self=False, force_symmetric=True)

    # instantiate dataset
    torch_dataset = ImputationDataset(
        *dataset.numpy(return_idx=True),
        training_mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        connectivity=adj,
        exogenous=exog_map,
        input_map=input_map,
        window=args.window,
        stride=args.stride,
    )

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    scalers = {"data": StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(
        torch_dataset, scalers=scalers, splitter=splitter, batch_size=args.batch_size // args.split_batch_in
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    additional_model_hparams = {
        "n_nodes": dm.n_nodes,
        "input_size": dm.n_channels,
        "u_size": 4,
        "output_size": dm.n_channels,
        "window_size": dm.window,
    }

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams}, target_cls=model_cls, return_dict=True
    )

    # loss and metrics
    loss_fn = MaskedMetric(
        metric_fn=getattr(torch.nn.functional, args.loss_fn), compute_on_step=True, metric_kwargs={"reduction": "none"}
    )

    metrics = {
        "mae": MaskedMAE(compute_on_step=False),
        "mse": MaskedMSE(compute_on_step=False),
        "mre": MaskedMRE(compute_on_step=False),
    }

    scheduler_class, scheduler_kwargs = get_scheduler(args.lr_scheduler, args)

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class, return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": args.lr, "weight_decay": args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        # scheduler_class=scheduler_class,
        # scheduler_kwargs=scheduler_kwargs,
        **imputer_kwargs,
    )

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor="val_mae", patience=args.patience, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor="val_mae", mode="min")

    tb_logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=logdir,
        logger=tb_logger,
        precision=args.precision,
        accumulate_grad_batches=args.split_batch_in,
        gpus=int(torch.cuda.is_available()),
        gradient_clip_val=args.grad_clip_val,
        limit_train_batches=args.batches_epoch * args.split_batch_in,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(
        imputer,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(batch_size=args.batch_inference),
    )

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)
    imputer.freeze()
    trainer.test(imputer, dataloaders=dm.test_dataloader(batch_size=args.batch_inference))

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader(batch_size=args.batch_inference))
    output = casting.numpy(output)
    y_hat, y_true, mask = output["y_hat"].squeeze(-1), output["y"].squeeze(-1), output["mask"].squeeze(-1)
    check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
    print(f"Test MAE: {check_mae:.2f}")
    return y_hat


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
