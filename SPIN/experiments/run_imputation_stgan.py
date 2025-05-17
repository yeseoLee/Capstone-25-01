import copy
import datetime
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
from spin.stgan_dataset import STGANBayDataset
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.nn.metrics import MaskedMAE, MaskedMetric, MaskedMRE, MaskedMSE
from tsl.utils import parser_utils
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


def get_dataset(dataset_name: str, root_dir=None):
    # STGAN 데이터셋 사용
    if dataset_name.endswith("_point"):
        p_fault, p_noise = 0.0, 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith("_block"):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")

    if dataset_name == "bay":
        # STGAN 데이터 형식의 Bay 데이터셋 로드
        stgan_dataset = STGANBayDataset(root_dir=root_dir).load()

        # 결측치를 직접 추가하는 로직 구현
        # 원본 데이터 복사
        data = stgan_dataset._target.copy()
        mask = np.ones_like(data, dtype=bool)

        # 시드 설정
        seed = 56789
        rng = np.random.RandomState(seed)

        # 데이터 크기
        time_steps, n_nodes, n_channels = data.shape

        # 포인트 결측치 추가 (p_noise 확률로 랜덤하게 데이터 포인트 마스킹)
        if p_noise > 0:
            noise_mask = rng.rand(time_steps, n_nodes, n_channels) < p_noise
            mask = mask & ~noise_mask

        # 블록 결측치 추가 (p_fault 확률로 각 노드의 연속된 블록 마스킹)
        if p_fault > 0:
            # 각 노드에 대해 독립적으로 처리
            for n in range(n_nodes):
                # p_fault 확률로 결측이 시작되는 시점들 결정
                fault_points = rng.rand(time_steps) < p_fault
                fault_indices = np.where(fault_points)[0]

                # 각 결측 시작점에 대해 min_seq에서 max_seq 사이의 랜덤한 길이로 마스킹
                min_seq, max_seq = 12, 12 * 4
                for idx in fault_indices:
                    if idx >= time_steps:
                        continue

                    # 랜덤 길이 결정
                    seq_len = rng.randint(min_seq, max_seq + 1)
                    end_idx = min(idx + seq_len, time_steps)

                    # 모든 채널에 대해 마스킹
                    mask[idx:end_idx, n, :] = False

        # 결측치 적용 (마스킹된 부분을 NaN으로 설정)
        masked_data = data.copy()
        masked_data[~mask] = np.nan

        # 데이터셋 업데이트
        stgan_dataset._target = masked_data
        stgan_dataset._mask = mask
        stgan_dataset._training_mask = mask.copy()
        stgan_dataset._eval_mask = mask.copy()

        return stgan_dataset

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
    parser.add_argument("--root-dir", type=str, default=None, help="STGAN 데이터셋 경로")

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


def run_experiment(args):  # noqa: C901
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, imputer_class = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name, root_dir=args.root_dir)

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
    time_emb = dataset.datetime_encoded(["day", "week"]).values()
    exog_map = {"global_temporal_encoding": time_emb}
    input_map = {"u": "temporal_encoding", "x": "data"}

    adj = dataset.get_connectivity(threshold=args.adj_threshold, include_self=False, force_symmetric=True)
    # PyTorch Geometric을 위한 edge_index 생성 (long 타입)
    rows, cols = np.where(adj > 0)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    # instantiate dataset
    torch_dataset = ImputationDataset(
        *dataset.numpy(return_idx=True),
        training_mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        connectivity=adj,  # 여기서는 adj 행렬 사용
        exogenous=exog_map,
        input_map=input_map,
        window=args.window,
        stride=args.stride,
    )

    # PyTorch Geometric을 위한 edge_index를 torch_dataset에 저장
    torch_dataset.edge_index = edge_index

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    # TSL 라이브러리의 SpatioTemporalDataModule 클래스는 splitter.split() 메서드를 기대합니다.
    # 반환된 splitter가 함수인 경우, 이를 처리하기 위한 래퍼 클래스 생성
    class SplitterWrapper:
        def __init__(self, split_fn):
            self.split_fn = split_fn
            self._splits = None

        def split(self, dataset):
            # 데이터셋의 인덱스를 가져오기
            idx = dataset.idx if hasattr(dataset, "idx") else np.arange(len(dataset))
            # 분할 함수 호출
            split_masks = self.split_fn(idx)
            # 각 데이터셋 유형에 해당하는 인덱스 저장
            self._splits = {k: np.where(v)[0] for k, v in split_masks.items()}
            return self._splits

        def get_split(self, split):
            if self._splits is None:
                raise RuntimeError("You must call split() before get_split()")
            return self._splits.get(split, None)

        @property
        def train_idxs(self):
            if self._splits is None:
                raise RuntimeError("You must call split() before accessing train_idxs")
            return self._splits.get("train", None)

        @property
        def val_idxs(self):
            if self._splits is None:
                raise RuntimeError("You must call split() before accessing val_idxs")
            return self._splits.get("val", None)

        @property
        def test_idxs(self):
            if self._splits is None:
                raise RuntimeError("You must call split() before accessing test_idxs")
            return self._splits.get("test", None)

    # splitter가 함수인 경우 래퍼로 감싸기
    if callable(splitter) and not hasattr(splitter, "split"):
        splitter = SplitterWrapper(splitter)

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
        "u_size": 31,  # STGAN time_features는 31개 (7일 + 24시간)
        "output_size": dm.n_channels,
        "window_size": dm.window,
        "h_size": dm.n_channels,  # 채널 수와 일치하도록 h_size 설정
        "z_size": dm.n_channels,  # 채널 수와 일치하도록 z_size 설정
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
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=args.batches_epoch if args.batches_epoch > 0 else 1.0,
    )

    trainer.fit(imputer, datamodule=dm)

    # load best model
    imputer.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    trainer.test(imputer, datamodule=dm)

    # Free memory
    del imputer, model_cls, trainer, dm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Parse args
    args = parse_args()
    # Run experiment
    run_experiment(args)
