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
from tsl.datasets.prototypes import Dataset
from tsl.nn.utils import casting
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


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--config", type=str, default="inference.yaml")
    parser.add_argument("--root", type=str, default="log")
    parser.add_argument("--root-dir", type=str, default=None, help="datasets/bay 데이터셋 경로")

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
        "u_size": 31,  # STGAN time_features는 31개 (7일 + 24시간)
        "output_size": dm.n_channels,
        "window_size": dm.window,
        # 추가로 확실하게 설정
        "h_size": dm.n_channels,  # 채널 수와 일치하도록 h_size 설정
        "z_size": dm.n_channels,  # 채널 수와 일치하도록 z_size 설정
        "support_stgan_format": True,  # STGAN 형식 지원 활성화
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

    # 데이터 크기
    time_steps, n_nodes, n_features, n_channels = dataset.shape

    # 마스크 초기화 (전체 True로 시작)
    mask = np.ones((time_steps, n_nodes, n_features, n_channels), dtype=bool)

    # 랜덤 생성기 초기화
    rng = np.random.RandomState(seed)

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
            min_seq, max_seq = 12, 36  # 원래 샘플 함수의 값 사용
            for idx in fault_indices:
                if idx >= time_steps:
                    continue

                # 랜덤 길이 결정
                seq_len = rng.randint(min_seq, max_seq + 1)
                end_idx = min(idx + seq_len, time_steps)

                # 모든 채널에 대해 마스킹
                mask[idx:end_idx, n, :, :] = False

    # 4D 마스크를 3D로 변환
    mask_reshaped = mask.reshape(time_steps, n_nodes, n_features * n_channels)

    # 평가 마스크 업데이트 (원본 4D 마스크는 dataset에 저장)
    dataset.set_eval_mask(mask)

    # 데이터 모듈 업데이트 (3D 마스크 사용)
    dm.torch_dataset.set_mask(dataset.training_mask.reshape(time_steps, n_nodes, n_features * n_channels))
    dm.torch_dataset.update_exogenous("eval_mask", mask_reshaped)


def run_experiment(args):  # noqa: C901
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

    dataset = get_dataset(exp_config["dataset_name"], root_dir=args.root_dir)

    ########################################
    # load data module                     #
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

    # instantiate dataset
    torch_dataset = ImputationDataset(
        data_reshaped,
        idx,
        training_mask=training_mask_reshaped,
        eval_mask=eval_mask_reshaped,
        connectivity=(edge_index, edge_weight),
        exogenous=exog_map,
        input_map=input_map,
        window=exp_config["window"],
        stride=exp_config["stride"],
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

    scalers = {"data": StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(
        torch_dataset, scalers=scalers, splitter=SplitterWrapper(splitter), batch_size=args.batch_size
    )
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
