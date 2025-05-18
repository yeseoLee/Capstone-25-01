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

    def __init__(self, root_dir=None, selected_nodes=None):
        """
        Args:
            root_dir: STGAN 데이터셋이 있는 디렉토리 경로
                기본값은 현재 작업 디렉토리의 '../datasets/bay'입니다.
            selected_nodes: 사용할 노드의 인덱스 리스트 (메모리 사용량 줄이기 위함)
                None이면 모든 노드 사용, 리스트이면 해당 인덱스의 노드만 사용
        """
        super().__init__()

        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets", "bay")

        self._root_dir = root_dir
        self._data_dir = os.path.join(root_dir, "data")
        self._exogenous = {}
        self.selected_nodes = selected_nodes

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
        full_data = np.load(data_path)
        self.time_features = np.loadtxt(time_features_path)
        full_node_adjacent = np.loadtxt(node_adjacent_path, dtype=np.int32)
        full_node_dist = np.loadtxt(node_dist_path)

        # 선택된 노드만 사용하는 경우
        if self.selected_nodes is not None:
            print(f"선택된 노드 {len(self.selected_nodes)}개만 사용합니다. (전체 {full_data.shape[1]}개 중)")
            self.data = full_data[:, self.selected_nodes, :, :]

            # 노드 인접 행렬 및 거리 행렬도 필터링
            self.node_adjacent = self._filter_adjacency(full_node_adjacent, self.selected_nodes)
            self.node_dist = full_node_dist[np.ix_(self.selected_nodes, self.selected_nodes)]
        else:
            self.data = full_data
            self.node_adjacent = full_node_adjacent
            self.node_dist = full_node_dist

        # STGAN 원본 형식 그대로 사용
        self._target = self.data

        # 선택된 노드에 기반한 인접 행렬 생성
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

    def _filter_adjacency(self, full_adjacency, selected_nodes):
        """전체 인접 리스트에서 선택된 노드만 포함하는 새로운 인접 리스트 생성

        Args:
            full_adjacency: 전체 노드의 인접 리스트
            selected_nodes: 선택된 노드 인덱스 리스트

        Returns:
            filtered_adjacency: 선택된 노드만 포함하는 인접 리스트
        """
        # 선택된 노드 인덱스를 매핑하는 딕셔너리 생성
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}

        # 새로운 인접 리스트 생성
        filtered_adjacency = np.zeros((len(selected_nodes), full_adjacency.shape[1]), dtype=np.int32)

        # 각 선택된 노드에 대해
        for new_idx, old_idx in enumerate(selected_nodes):
            # 원래 인접 리스트 가져오기
            neighbors = full_adjacency[old_idx]

            # 선택된 노드에 속하는 이웃만 새로운 인덱스로 변환
            filtered_neighbors = []
            for neighbor in neighbors:
                if neighbor in node_mapping:
                    filtered_neighbors.append(node_mapping[neighbor])
                else:
                    # 선택되지 않은 이웃은 -1로 표시 (나중에 제거)
                    filtered_neighbors.append(-1)

            # 필터링된 인접 리스트 저장
            filtered_adjacency[new_idx, : len(filtered_neighbors)] = filtered_neighbors

        return filtered_adjacency

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


def get_dataset(dataset_name: str, root_dir=None, selected_nodes=None):
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
        # 원본 STGAN 데이터 형식을 직접 로드 (선택된 노드만 사용하는 옵션 추가)
        stgan_dataset = DirectSTGANDataset(root_dir=root_dir, selected_nodes=selected_nodes).load()

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

    # 노드 선택 관련 인수 추가
    parser.add_argument("--use-node-subset", action="store_true", help="일부 노드만 사용하여 메모리 사용량 감소")
    parser.add_argument("--node-ratio", type=float, default=0.5, help="전체 노드 중 사용할 비율 (0.0-1.0)")
    parser.add_argument("--node-list", type=str, default=None, help="사용할 노드 인덱스 목록 (쉼표로 구분)")

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

    # 설정 파일이 있는 경우 해당 내용으로 덮어쓰기
    if args.config:
        if not os.path.isfile(args.config):
            cfg_path = os.path.join(config.config_dir, args.config)
            with open(cfg_path, "r") as fp:
                config_args = yaml.load(fp, Loader=yaml.FullLoader)
            for arg in config_args:
                setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):  # noqa: C901
    # 시작 시간 기록
    import os as os_util
    import time

    import psutil

    start_time = time.time()
    process = psutil.Process(os_util.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB 단위

    logger.info(f"실험 시작 - 초기 메모리 사용량: {initial_memory:.2f} MB")

    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, imputer_class = get_model_classes(args.model_name)

    # 노드 서브셋 옵션 처리
    selected_nodes = None
    if args.use_node_subset:
        # 지정된 노드 목록이 있는 경우
        if args.node_list:
            selected_nodes = [int(node) for node in args.node_list.split(",")]
            logger.info(f"사용자 지정 노드 {len(selected_nodes)}개를 사용합니다: {selected_nodes}")
        else:
            # 임시로 모든 노드를 로드하여 전체 노드 수 확인
            temp_dataset = get_dataset(args.dataset_name, root_dir=args.root_dir)
            total_nodes = temp_dataset.data.shape[1]

            # 비율에 따라 랜덤하게 노드 선택
            num_nodes = max(1, int(total_nodes * args.node_ratio))
            np.random.seed(args.seed)  # 동일한 시드 사용
            selected_nodes = np.random.choice(total_nodes, num_nodes, replace=False).tolist()
            logger.info(f"전체 {total_nodes}개 노드 중 {num_nodes}개({args.node_ratio*100:.1f}%)를 랜덤하게 선택했습니다.")

    # 선택된 노드로 데이터셋 로드
    dataset = get_dataset(args.dataset_name, root_dir=args.root_dir, selected_nodes=selected_nodes)

    # 데이터셋 로드 후 메모리 사용량 측정
    dataset_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
    logger.info(f"데이터셋 로드 완료 - 메모리 사용량: {dataset_memory:.2f} MB (증가: {dataset_memory - initial_memory:.2f} MB)")

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

    # 4D -> 3D 변환: [시간, 노드, 특성, 채널] -> [시간, 노드, 특성*채널]
    time_steps, n_nodes, n_features, n_channels = data.shape
    data_reshaped = data.reshape(time_steps, n_nodes, n_features * n_channels)

    # 마스크도 같은 방식으로 변환
    training_mask_reshaped = dataset.training_mask.reshape(time_steps, n_nodes, n_features * n_channels)
    eval_mask_reshaped = dataset.eval_mask.reshape(time_steps, n_nodes, n_features * n_channels)

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

    # 실험 완료 후 메모리 사용량 및 실행 시간 측정
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
    total_time = end_time - start_time

    logger.info("=" * 50)
    logger.info("실험 완료 - 성능 요약")
    logger.info(f"총 실행 시간: {total_time:.2f} 초 ({total_time/60:.2f} 분)")
    logger.info(f"최종 메모리 사용량: {final_memory:.2f} MB")
    logger.info(f"메모리 증가량: {final_memory - initial_memory:.2f} MB")

    if args.use_node_subset:
        # 선택된 노드 정보
        if args.node_list:
            node_info = f"사용자 지정 노드 {len(selected_nodes)}개"
        else:
            node_info = f"전체의 {args.node_ratio*100:.1f}% ({len(selected_nodes)}개 노드)"
        logger.info(f"노드 서브셋 사용: {node_info}")
    else:
        logger.info("전체 노드 사용")
    logger.info("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
