import copy
import os

import numpy as np
import pytorch_lightning as pl
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
import torch
import tsl
from tsl import config
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import PemsBay
from tsl.nn.utils import casting
from tsl.ops.imputation import add_missing_values, sample_mask
from tsl.utils import ArgParser, numpy_metrics, parser_utils
from tsl.utils.python_utils import ensure_list
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


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--config", type=str, default="inference.yaml")
    parser.add_argument("--root", type=str, default="log")

    # Data sparsity params
    parser.add_argument("--p-fault", type=float, default=0.0)
    parser.add_argument("--p-noise", type=float, default=0.75)
    parser.add_argument("--test-mask-seed", type=int, default=None)

    # Splitting/aggregation params
    parser.add_argument("--val-len", type=float, default=0.1)
    parser.add_argument("--test-len", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, "r") as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def load_model(exp_dir, exp_config, dm):
    model_cls, imputer_class = get_model_classes(exp_config["model_name"])
    additional_model_hparams = {
        "n_nodes": dm.n_nodes,
        "input_size": dm.n_channels,
        "u_size": 4,
        "output_size": dm.n_channels,
        "window_size": dm.window,
    }

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**exp_config, **additional_model_hparams}, target_cls=model_cls, return_dict=True
    )

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(exp_config, imputer_class, return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={},
        loss_fn=None,
        **imputer_kwargs,
    )

    model_path = None
    for file in os.listdir(exp_dir):
        if file.endswith(".ckpt"):
            model_path = os.path.join(exp_dir, file)
            break
    if model_path is None:
        raise ValueError("Model not found.")

    imputer.load_model(model_path)
    imputer.freeze()
    return imputer


def update_test_eval_mask(dm, dataset, p_fault, p_noise, seed=None):
    if seed is None:
        seed = np.random.randint(1e9)
    random = np.random.default_rng(seed)
    dataset.set_eval_mask(sample_mask(dataset.shape, p=p_fault, p_noise=p_noise, min_seq=12, max_seq=36, rng=random))
    dm.torch_dataset.set_mask(dataset.training_mask)
    dm.torch_dataset.update_exogenous("eval_mask", dataset.eval_mask)


def run_experiment(args):
    # Set configuration
    args = copy.deepcopy(args)
    tsl.logger.disabled = True

    ########################################
    # load config                          #
    ########################################

    if args.root is None:
        root = tsl.config.log_dir
    else:
        root = os.path.join(tsl.config.curr_dir, args.root)
    exp_dir = os.path.join(root, args.dataset_name, args.model_name, args.exp_name)

    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        exp_config = yaml.load(fp, Loader=yaml.FullLoader)

    ########################################
    # load dataset                         #
    ########################################

    dataset = get_dataset(exp_config["dataset_name"])

    ########################################
    # load data module                     #
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
        window=exp_config["window"],
        stride=exp_config["stride"],
    )

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    scalers = {"data": StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset, scalers=scalers, splitter=splitter, batch_size=args.batch_size)
    dm.setup()

    ########################################
    # load model                           #
    ########################################

    imputer = load_model(exp_dir, exp_config, dm)

    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()))

    ########################################
    # inference                            #
    ########################################

    seeds = ensure_list(args.test_mask_seed)
    mae = []

    for seed in seeds:
        # Change evaluation mask
        update_test_eval_mask(dm, dataset, args.p_fault, args.p_noise, seed)

        output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
        output = casting.numpy(output)
        y_hat, y_true, mask = output["y_hat"].squeeze(-1), output["y"].squeeze(-1), output["mask"].squeeze(-1)

        check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
        mae.append(check_mae)
        print(f"SEED {seed} - Test MAE: {check_mae:.2f}")

    print(f"MAE over {len(seeds)} runs: {np.mean(mae):.2f}±{np.std(mae):.2f}")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
