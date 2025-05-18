import copy
import datetime
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
from spin.scheduler import CosineSchedulerWithRestarts
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tsl import config, logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets.prototypes import Dataset
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


# STGAN 데이터셋을 직접 로드하는 클래스 구현
class DirectSTGANDataset(Dataset):
    """STGAN 데이터셋을 직접 로드하는 클래스입니다.
    데이터 형식 변환 없이 원본 형태로 사용합니다.
    """

    def __init__(self, root_dir=None):
        """
        Args:
            root_dir: STGAN 데이터셋이 있는 디렉토리 경로
                기본값은 현재 작업 디렉토리의 '../datasets/bay'입니다.
        """
        super().__init__()

        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets", "bay")

        self._root_dir = root_dir
        self._data_dir = os.path.join(root_dir, "data")
        self._exogenous = {}

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def shape(self):
        """데이터셋의 형태 반환"""
        if hasattr(self, "_target"):
            return self._target.shape
        else:
            return None

    @property
    def n_nodes(self):
        """노드 수 반환"""
        if hasattr(self, "_target"):
            return self._target.shape[1]
        else:
            return None

    @property
    def n_channels(self):
        """채널 수 반환"""
        if hasattr(self, "_target"):
            # 원본 STGAN 형식 유지: (시간, 노드, 특성, 채널)
            return self._target.shape[2] * self._target.shape[3]
        else:
            return None

    @property
    def length(self):
        """데이터셋의 길이(시간 차원) 반환"""
        if hasattr(self, "_target"):
            return self._target.shape[0]
        else:
            return None

    @property
    def training_mask(self):
        """훈련 마스크 반환"""
        return self._training_mask

    @property
    def eval_mask(self):
        """평가 마스크 반환"""
        return self._eval_mask

    def load(self):
        # 데이터 로드
        data_path = os.path.join(self.data_dir, "data.npy")
        time_features_path = os.path.join(self.data_dir, "time_features.txt")
        node_adjacent_path = os.path.join(self.data_dir, "node_adjacent.txt")
        node_dist_path = os.path.join(self.data_dir, "node_dist.txt")

        # 교통 데이터 로드 (time, node, feature, channel)
        self.data = np.load(data_path)
        self.time_features = np.loadtxt(time_features_path)
        self.node_adjacent = np.loadtxt(node_adjacent_path, dtype=np.int32)
        self.node_dist = np.loadtxt(node_dist_path)

        # STGAN 원본 형식 그대로 사용
        self._target = self.data

        # 인접 행렬 생성
        n_nodes = self.data.shape[1]
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j, neighbor in enumerate(self.node_adjacent[i]):
                if j > 0:  # 첫 번째는 노드 자신일 수 있음
                    adj_matrix[i, neighbor] = 1

        # 인접 행렬의 에지 인덱스 형식으로 직접 변환
        rows, cols = np.where(adj_matrix > 0)
        edge_index = np.array([rows, cols], dtype=np.int64)
        edge_weight = adj_matrix[rows, cols]

        # 속성 직접 설정
        self._connectivity = adj_matrix
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self._mask = np.ones_like(self.data, dtype=bool)
        self._training_mask = self._mask.copy()
        self._eval_mask = self._mask.copy()

        # 시간 정보 추가
        self.add_exogenous("temporal_encoding", self.time_features)

        # Dataset 클래스의 필수 속성 설정
        self._idx = np.arange(self.data.shape[0])
        self._t = np.arange(self.data.shape[0])

        return self

    def add_exogenous(self, name, data, axis=0):
        """외부 데이터 추가"""
        if isinstance(data, list):
            data = np.array(data)
        if axis == 0 and len(data) != self._target.shape[0]:
            raise ValueError(f"데이터 길이 불일치: {len(data)} vs {self._target.shape[0]}")
        self._exogenous[name] = (data, axis)
        return self

    def get_exogenous(self, name):
        """저장된 외부 데이터 반환"""
        if name not in self._exogenous:
            raise KeyError(f"외부 데이터 '{name}'이(가) 없습니다.")
        return self._exogenous[name][0]

    def set_eval_mask(self, mask):
        """평가 마스크 설정"""
        self._eval_mask = mask.copy()

    def datetime_encoded(self, fields=None):
        """요일 및 시간을 인코딩한 특성 반환"""
        result = {}
        if fields is None or "day" in fields:
            # 시간대 정보 (24 시간)
            result["day"] = self.time_features[:, 7:]
        if fields is None or "week" in fields:
            # 요일 정보 (7일)
            result["week"] = self.time_features[:, :7]

        # NumPy 배열을 반환하는 대신 TSL에서 사용하는 형식에 맞게 반환
        class EncodedDatetime:
            def __init__(self, data):
                self.data = data

            def values(self):
                # 데이터를 단일 배열로 결합
                all_features = np.concatenate(list(self.data.values()), axis=1)
                return all_features

        return EncodedDatetime(result)

    def get_connectivity(self, threshold=None, include_self=False, force_symmetric=False):
        """연결 행렬 반환"""
        # 인접 행렬 복사
        adj_matrix = self._connectivity.copy()

        # 자기 연결 포함 여부
        if include_self:
            adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

        # 대칭 행렬 강제
        if force_symmetric:
            adj_matrix = adj_matrix + adj_matrix.T
            adj_matrix = (adj_matrix > 0).astype(float)

        # 임계값 적용
        if threshold is not None:
            adj_matrix[adj_matrix < threshold] = 0

        # 에지 인덱스와 가중치 반환
        rows, cols = np.where(adj_matrix > 0)
        edge_index = np.array([rows, cols], dtype=np.int64)
        edge_weight = adj_matrix[rows, cols]

        return edge_index, edge_weight

    def numpy(self, return_idx=False):
        """NumPy 배열 반환"""
        if return_idx:
            return self._target, self._idx
        return self._target

    def get_splitter(self, method="temporal", **kwargs):
        """데이터 분할기 반환"""

        def splitter(idx, **kwargs):
            # 시간 기반 분할
            if method == "temporal":
                val_len = kwargs.get("val_len", 0.1)
                test_len = kwargs.get("test_len", 0.2)

                if isinstance(val_len, float):
                    val_len = int(len(idx) * val_len)
                if isinstance(test_len, float):
                    test_len = int(len(idx) * test_len)

                test_start = len(idx) - test_len
                val_start = test_start - val_len

                train_idx = idx[:val_start]
                val_idx = idx[val_start:test_start]
                test_idx = idx[test_start:]

                return train_idx, val_idx, test_idx
            else:
                raise ValueError(f"지원하지 않는 분할 방법: {method}")

        return splitter


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
        # 원본 STGAN 데이터 형식을 직접 로드
        stgan_dataset = DirectSTGANDataset(root_dir=root_dir).load()

        # 결측치를 직접 추가하는 로직 구현
        # 원본 데이터 복사
        data = stgan_dataset._target.copy()
        mask = np.ones_like(data, dtype=bool)

        # 시드 설정
        seed = 56789
        rng = np.random.RandomState(seed)

        # 데이터 크기
        time_steps, n_nodes, n_features, n_channels = data.shape

        # 포인트 결측치 추가 (p_noise 확률로 랜덤하게 데이터 포인트 마스킹)
        if p_noise > 0:
            noise_mask = rng.rand(time_steps, n_nodes, n_features, n_channels) < p_noise
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

                    # 모든 채널과 특성에 대해 마스킹
                    mask[idx:end_idx, n, :, :] = False

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
    """스케줄러 설정을 가져옵니다.

    Args:
        scheduler_name: 스케줄러 이름 (None, 'cosine', 'cosine_warm_restarts', 'cosine_with_restarts')
        args: 명령줄 매개변수

    Returns:
        PyTorch Lightning 형식의 스케줄러 설정 딕셔너리
    """
    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        scheduler = lambda opt: CosineAnnealingLR(optimizer=opt, T_max=args.epochs, eta_min=0.1 * args.lr)
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
    elif scheduler_name == "cosine_warm_restarts":
        scheduler = lambda opt: CosineAnnealingWarmRestarts(
            optimizer=opt,
            T_0=args.epochs // 3,  # 첫 번째 재시작까지의 에포크 수
            T_mult=2,  # 이후 주기는 2배씩 증가
            eta_min=0.1 * args.lr,
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
    elif scheduler_name == "cosine_with_restarts":
        # spin.scheduler의 CosineSchedulerWithRestarts 사용
        scheduler = lambda opt: CosineSchedulerWithRestarts(
            optimizer=opt,
            num_warmup_steps=args.epochs // 10,  # 웜업 단계 수
            num_training_steps=args.epochs,  # 전체 학습 단계 수
            min_factor=0.1,  # 최소 학습률 비율
            linear_decay=0.67,  # 선형 감소 비율
            num_cycles=3,  # 재시작 횟수
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
    else:
        raise ValueError(f"지원하지 않는 스케줄러: {scheduler_name}")


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="spin_h")
    parser.add_argument("--dataset-name", type=str, default="bay_block")
    parser.add_argument("--config", type=str, default="imputation/spin_h.yaml")
    parser.add_argument("--root-dir", type=str, default=None, help="datasets/bay 데이터셋 경로")

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
    edge_index = torch.tensor(adj[0], dtype=torch.long)
    edge_weight = torch.tensor(adj[1], dtype=torch.float) if adj[1] is not None else None

    # 4D 데이터를 3D로 변환
    data, idx = dataset.numpy(return_idx=True)
    # 데이터 형상 출력 (디버깅용)
    print(f"원본 데이터 형상: {data.shape}")

    # 4D -> 3D 변환: [시간, 노드, 특성, 채널] -> [시간, 노드, 특성*채널]
    time_steps, n_nodes, n_features, n_channels = data.shape
    data_reshaped = data.reshape(time_steps, n_nodes, n_features * n_channels)

    # 마스크도 같은 방식으로 변환
    training_mask_reshaped = dataset.training_mask.reshape(time_steps, n_nodes, n_features * n_channels)
    eval_mask_reshaped = dataset.eval_mask.reshape(time_steps, n_nodes, n_features * n_channels)

    print(f"변환된 데이터 형상: {data_reshaped.shape}")

    # 데이터 차원 분석을 위한 추가 정보 출력
    real_input_size = n_features * n_channels
    print(f"실제 입력 채널 수: {real_input_size}")
    print(f"배치 차원 정보: 윈도우 크기={args.window}, 노드 수={n_nodes}, 특성*채널={real_input_size}")

    # instantiate dataset
    torch_dataset = ImputationDataset(
        data_reshaped,
        idx,
        training_mask=training_mask_reshaped,
        eval_mask=eval_mask_reshaped,
        connectivity=(edge_index, edge_weight),
        exogenous=exog_map,
        input_map=input_map,
        window=args.window,
        stride=args.stride,
    )

    # get train/val/test indices with dataset's own splitter
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    # Wrapper for splitter to match TSL library's expectations
    class SplitterWrapper:
        def __init__(self, split_fn):
            self.split_fn = split_fn
            self._idxs = None

        def split(self, dataset):
            # 데이터셋의 인덱스를 가져오기
            if hasattr(dataset, "_idx"):
                idx = dataset._idx
            else:
                idx = np.arange(len(dataset))

            # 분할 함수를 사용하여 인덱스 분할
            self._idxs = self.split_fn(idx, val_len=args.val_len, test_len=args.test_len)
            return self._idxs

        def get_split(self, split):
            if self._idxs is None:
                raise ValueError("먼저 split 메서드를 호출해야 합니다.")
            return self._idxs[split]

        @property
        def train_idxs(self):
            if self._idxs is None:
                raise ValueError("먼저 split 메서드를 호출해야 합니다.")
            return self._idxs[0]

        @property
        def val_idxs(self):
            if self._idxs is None:
                raise ValueError("먼저 split 메서드를 호출해야 합니다.")
            return self._idxs[1]

        @property
        def test_idxs(self):
            if self._idxs is None:
                raise ValueError("먼저 split 메서드를 호출해야 합니다.")
            return self._idxs[2]

    # StandardScaler 초기화 (axis=(0, 1)는 시간과 노드 차원에 대해 표준화)
    scalers = {"data": StandardScaler(axis=(0, 1))}

    # 데이터 모듈 초기화
    dm = SpatioTemporalDataModule(
        torch_dataset,
        scalers=scalers,
        splitter=SplitterWrapper(splitter),
        batch_size=args.batch_size // args.split_batch_in,
        workers=args.workers,
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    # 모델 차원을 명시적으로 설정
    # STGAN 데이터셋의 특성*채널이 12임을 반영하여 h_size와 z_size를 12로 강제 설정
    h_size = 12  # 매개변수에 관계없이 12로 고정
    z_size = 12  # 매개변수에 관계없이 12로 고정

    print(f"강제 설정된 모델 크기 - h_size: {h_size}, z_size: {z_size}")

    additional_model_hparams = {
        "n_nodes": dm.n_nodes,
        "input_size": dm.n_channels,
        "u_size": 31,  # STGAN time_features는 31개 (7일 + 24시간)
        "output_size": dm.n_channels,
        "window_size": dm.window,
        # 모델 차원을 명시적으로 설정
        "h_size": h_size,
        "z_size": z_size,
        "support_stgan_format": True,  # STGAN 형식 지원 활성화
    }

    # 모델 파라미터 로깅
    print("모델 구성 파라미터:")
    print(f"  - input_size: {dm.n_channels}")
    print(f"  - h_size: {h_size}")
    print(f"  - z_size: {z_size}")
    print(f"  - n_nodes: {dm.n_nodes}")
    print(f"  - window_size: {dm.window}")

    # model's inputs
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams}, target_cls=model_cls, return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn), compute_on_step=True, metric_kwargs={"reduction": "none"})

    metrics = {
        "mae": MaskedMAE(compute_on_step=False),
        "mse": MaskedMSE(compute_on_step=False),
        "mre": MaskedMRE(compute_on_step=False),
    }

    # 옵티마이저 설정
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class, return_dict=True)

    # 스케줄러가 필요한 경우 PyTorch Lightning 콜백으로 추가
    if args.lr_scheduler is not None:
        # 스케줄러 설정 가져오기
        scheduler_config = get_scheduler(args.lr_scheduler, args)

        # imputer 초기화
        imputer = imputer_class(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs={"lr": args.lr, "weight_decay": args.l2_reg},
            loss_fn=loss_fn,
            metrics=metrics,
            **imputer_kwargs,
        )

        # configure_optimizers 메서드 오버라이드
        original_configure_optimizers = imputer.configure_optimizers

        def new_configure_optimizers():
            # 기존 메서드로부터 옵티마이저 가져오기
            optimizers = original_configure_optimizers()

            # 옵티마이저가 리스트인 경우 첫 번째 항목 사용
            if isinstance(optimizers, list) and len(optimizers) > 0:
                optimizer = optimizers[0]
            # 옵티마이저가 단일 항목인 경우 그대로 사용
            elif not isinstance(optimizers, dict):
                optimizer = optimizers
            # 딕셔너리인 경우 'optimizer' 키 사용
            elif isinstance(optimizers, dict) and "optimizer" in optimizers:
                optimizer = optimizers["optimizer"]
                # 기존 lr_scheduler가 있는 경우 업데이트
                if "lr_scheduler" in optimizers:
                    optimizers["lr_scheduler"] = scheduler_config
                    return optimizers
            else:
                # 예상치 못한 형식인 경우 새 옵티마이저 생성
                optimizer = torch.optim.Adam(imputer.parameters(), lr=args.lr, weight_decay=args.l2_reg)

            # 스케줄러 함수가 람다 함수인 경우 적용
            if callable(scheduler_config["scheduler"]):
                scheduler_dict = {
                    "scheduler": scheduler_config["scheduler"](optimizer),
                    "interval": scheduler_config["interval"],
                    "frequency": scheduler_config["frequency"],
                }
                return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
            else:
                return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        # 메서드 교체
        imputer.configure_optimizers = new_configure_optimizers
    else:
        # 스케줄러 없이 imputer 초기화
        imputer = imputer_class(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs={"lr": args.lr, "weight_decay": args.l2_reg},
            loss_fn=loss_fn,
            metrics=metrics,
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


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
