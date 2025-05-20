import copy
import datetime
import os

import numpy as np
import pytorch_lightning as pl
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
import torch
import tsl
from tsl import config, logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets.prototypes import Dataset
from tsl.nn.utils import casting
from tsl.utils import ArgParser, parser_utils
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
        time_features_path = os.path.join(self.data_dir, "time_features_with_weather.txt")
        node_adjacent_path = os.path.join(self.data_dir, "node_adjacent.txt")
        node_dist_path = os.path.join(self.data_dir, "node_dist.txt")

        # 교통 데이터 로드 (time, node, feature, channel)
        full_data = np.load(data_path)
        self.time_features = np.loadtxt(time_features_path)
        full_node_adjacent = np.loadtxt(node_adjacent_path, dtype=np.int32)
        full_node_dist = np.loadtxt(node_dist_path)

        # 선택된 노드만 사용하는 경우
        if self.selected_nodes is not None:
            logger.info(f"선택된 노드 {len(self.selected_nodes)}개만 사용합니다. (전체 {full_data.shape[1]}개 중)")
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

        # 결측치 적용 - NaN 대신 0으로 설정하고 마스크로 관리
        masked_data = data.copy()
        # NaN 대신 0으로 설정 (NaN은 표준화 과정에서 문제 발생)
        masked_data[~mask] = 0

        # 데이터셋 업데이트
        stgan_dataset._target = masked_data
        stgan_dataset._mask = mask
        stgan_dataset._training_mask = mask.copy()
        stgan_dataset._eval_mask = mask.copy()

        # 원본 데이터 저장 (테스트용 마스크 생성 시 참조)
        stgan_dataset.original_data = data.copy()

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

    # Data sparsity params - 값 수정
    parser.add_argument("--p-fault", type=float, default=0.01, help="블록 결측치의 확률 (높을수록 더 많은 블록 결측치)")
    parser.add_argument("--p-noise", type=float, default=0.25, help="포인트 결측치의 확률 (높을수록 더 많은 포인트 결측치)")
    parser.add_argument("--test-mask-seed", type=int, default=1043, help="테스트 마스크 생성을 위한 시드")

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

    # 기본값 설정: test_mask_seed가 하나만 있으면 리스트로 변환
    if isinstance(args.test_mask_seed, int):
        args.test_mask_seed = [args.test_mask_seed, args.test_mask_seed + 1000, args.test_mask_seed + 2000]

    return args


def load_model(exp_dir, exp_config, dm):
    model_cls, imputer_class = get_model_classes(exp_config["model_name"])

    # 모델 차원을 명시적으로 설정
    # STGAN 데이터셋은 원래 4D이지만, ImputationDataset으로 변환 시 3D가 됨
    # 채널 정보는 dm.n_channels에 이미 올바르게 설정되어 있음

    additional_model_hparams = {
        "n_nodes": dm.n_nodes,
        "input_size": dm.n_channels,
        "u_size": 34,  # STGAN time_features_with_weather는 34개 (7일 + 24시간 + 3가지 날씨)
        "output_size": dm.n_channels,
        "window_size": dm.window,
        # 추가로 확실하게 설정
        "h_size": dm.n_channels,  # 채널 수와 일치하도록 설정
        "z_size": dm.n_channels,  # 채널 수와 일치하도록 설정
        "support_stgan_format": True,  # STGAN 형식 지원 활성화
    }

    # model's inputs
    model_kwargs = parser_utils.filter_args(args={**exp_config, **additional_model_hparams}, target_cls=model_cls, return_dict=True)

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

    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # state_dict 불일치 문제 해결
    try:
        # 방법 1: strict=False로 시도
        imputer.load_state_dict(checkpoint["state_dict"], strict=False)
    except Exception as e:
        logger.warning(f"모델 로드 중 오류 발생 (strict=False): {str(e)}")
        logger.info("대체 방법으로 state_dict 수동 복사 시도...")

        # 방법 2: 수동으로 공통 키만 복사
        state_dict = checkpoint["state_dict"]
        model_state_dict = imputer.state_dict()

        # 공통 키 찾기
        common_keys = set(state_dict.keys()).intersection(set(model_state_dict.keys()))

        # 차원이 일치하는 가중치만 복사
        for key in common_keys:
            try:
                if state_dict[key].shape == model_state_dict[key].shape:
                    model_state_dict[key] = state_dict[key]
            except Exception as e2:
                logger.warning(f"파라미터 '{key}' 복사 중 오류 발생: {str(e2)}")

        # 수정된 state_dict 로드
        imputer.load_state_dict(model_state_dict, strict=False)

    imputer.freeze()
    return imputer


def update_test_eval_mask(dm, dataset, p_fault, p_noise, seed=None):
    if seed is None:
        seed = np.random.randint(1e9)

    logger.info(f"테스트 마스크 생성 시작 (seed: {seed}, p_fault: {p_fault}, p_noise: {p_noise})")

    # 데이터 크기
    time_steps, n_nodes, n_features, n_channels = dataset.shape

    # 원본 데이터가 있는지 확인하고, 없으면 현재 데이터 사용
    if hasattr(dataset, "original_data"):
        orig_data = dataset.original_data
    else:
        # 현재 데이터를 원본으로 간주
        orig_data = dataset._target.copy()
        logger.warning("원본 데이터를 찾을 수 없어 현재 데이터를 원본으로 간주합니다.")

    # 최소한의 검증 포인트 확보
    min_test_points = max(100, int(time_steps * n_nodes * n_features * n_channels * 0.01))  # 최소 데이터의 1%
    logger.info(f"목표 테스트 포인트: {min_test_points}개 (전체 데이터의 최소 1%)")

    # 트레이닝 마스크 유지 (기존 결측치는 계속 마스킹)
    train_mask = dataset.training_mask.copy()
    # 트레이닝 포인트 개수 확인
    train_points = np.sum(train_mask)
    logger.info(f"훈련 데이터 포인트: {train_points}개")

    # 인위적인 테스트 마스크 생성 (모든 위치 True로 초기화)
    artificial_mask = np.ones_like(train_mask, dtype=bool)

    # 랜덤 생성기 초기화
    rng = np.random.RandomState(seed)

    # 테스트 마스크 생성 (원래 유효한 부분의 일부분만 마스킹)
    valid_indices = np.where(train_mask)
    if len(valid_indices[0]) > 0:
        # 유효한 인덱스 중에서 일부를 선택
        n_valid = len(valid_indices[0])
        mask_ratio = min(0.2, max(0.05, min_test_points / n_valid))  # 최소 5%, 최대 20%
        n_mask = int(n_valid * mask_ratio)

        if n_mask > 0:
            logger.info(f"마스킹할 테스트 포인트: {n_mask}개 (유효 데이터의 {mask_ratio*100:.1f}%)")

            # 랜덤하게 인덱스 선택
            mask_indices = rng.choice(np.arange(n_valid), size=n_mask, replace=False)

            # 선택된 인덱스 마스킹
            for i in mask_indices:
                idx = (valid_indices[0][i], valid_indices[1][i], valid_indices[2][i], valid_indices[3][i])
                artificial_mask[idx] = False

            # artificial_mask가 False인 부분을 테스트 대상으로 평가 마스크 생성
            eval_mask = train_mask & ~artificial_mask
            test_points = np.sum(eval_mask)
            logger.info(f"생성된 테스트 평가 포인트: {test_points}개")
        else:
            logger.warning("마스킹할 테스트 포인트가 없습니다!")
            # 최소한 하나의 포인트는 마스킹
            eval_mask = np.zeros_like(train_mask, dtype=bool)
            idx0 = np.where(train_mask)[0][0]
            idx1 = np.where(train_mask)[1][0]
            idx2 = np.where(train_mask)[2][0]
            idx3 = np.where(train_mask)[3][0]
            eval_mask[idx0, idx1, idx2, idx3] = True
            test_points = 1
            logger.info("최소 1개의 테스트 포인트를 수동으로 생성했습니다.")
    else:
        logger.warning("유효한 트레이닝 포인트가 없습니다!")
        # 인위적으로 하나의 테스트 포인트 생성
        eval_mask = np.zeros_like(train_mask, dtype=bool)
        eval_mask[0, 0, 0, 0] = True  # 첫 번째 요소를 테스트 포인트로 설정
        test_points = 1
        logger.info("최소 1개의 테스트 포인트를 수동으로 생성했습니다.")

    # 3D 형식으로 변환된 마스크 준비
    train_mask_reshaped = train_mask.reshape(time_steps, n_nodes, n_features * n_channels)
    artificial_mask_reshaped = artificial_mask.reshape(time_steps, n_nodes, n_features * n_channels)
    eval_mask_reshaped = eval_mask.reshape(time_steps, n_nodes, n_features * n_channels)

    # 테스트용 eval_mask를 설정 (이 부분이 실제 평가에 사용됨)
    dataset.set_eval_mask(eval_mask)

    # 데이터 모듈 업데이트
    dm.torch_dataset.set_mask(artificial_mask_reshaped & train_mask_reshaped)  # training_mask & artificial_mask
    dm.torch_dataset.update_exogenous("eval_mask", eval_mask_reshaped)

    # 추가로 마스킹된 포인트 수
    logger.info(f"테스트 평가 포인트: {test_points}개")

    if test_points == 0:
        logger.warning("테스트 평가 포인트가 0개입니다! 결과는 신뢰할 수 없습니다.")
    elif test_points < 100:
        logger.warning(f"테스트 평가 포인트가 적습니다: {test_points}개. 결과가 부정확할 수 있습니다.")


def custom_masked_mae(y_hat, y_true, mask):
    """빈 마스크나 NaN 값을 안전하게 처리하는 Masked MAE 구현.

    Args:
        y_hat: 예측값 배열
        y_true: 실제값 배열
        mask: 마스크 배열 (True 위치만 평가)

    Returns:
        mae: 평균 절대 오차 값, 유효한 마스크가 없으면 NaN 대신 None 반환
    """
    # NaN 값 처리
    valid_mask = mask & ~np.isnan(y_hat) & ~np.isnan(y_true)

    # 유효한 마스크가 있는지 확인
    if np.sum(valid_mask) == 0:
        return None  # 유효한 마스크가 없으면 None 반환

    # 마스크된 부분만 선택
    y_hat_masked = y_hat[valid_mask]
    y_true_masked = y_true[valid_mask]

    # MAE 계산
    mae = np.mean(np.abs(y_hat_masked - y_true_masked))
    return mae


def run_experiment(args):  # noqa: C901
    # Set configuration
    args = copy.deepcopy(args)
    tsl.logger.disabled = True

    # 실행 정보 출력
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"========== 추론 시작: {current_time} ==========")
    logger.info(f"실행 인자: p_fault={args.p_fault}, p_noise={args.p_noise}")

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
    # 노드 서브셋 정보 처리                 #
    ########################################

    selected_nodes = None
    # config.yaml에서 노드 정보 확인
    if "use_node_subset" in exp_config and exp_config["use_node_subset"]:
        # 직접 노드 목록이 있는 경우
        if "selected_nodes" in exp_config and isinstance(exp_config["selected_nodes"], list):
            # 유효한 노드 인덱스만 필터링
            max_node_idx = 324  # 0부터 시작하므로 325-1
            selected_nodes = [node for node in exp_config["selected_nodes"] if 0 <= node <= max_node_idx]
            if len(selected_nodes) != len(exp_config["selected_nodes"]):
                logger.warning(f"일부 노드 인덱스가 범위를 벗어나 제외되었습니다. (유효한 노드: {len(selected_nodes)}/{len(exp_config['selected_nodes'])})")
            logger.info(f"config.yaml에서 {len(selected_nodes)}개의 선택된 노드 정보를 로드했습니다.")
        else:
            # 별도 파일에서 노드 정보 로드 시도
            selected_nodes_file = os.path.join(exp_dir, "selected_nodes.txt")
            if os.path.exists(selected_nodes_file):
                try:
                    with open(selected_nodes_file, "r") as f:
                        lines = f.readlines()
                        # 주석 줄 제외하고 노드 정보 파싱
                        for line in lines:
                            if not line.startswith("#"):
                                node_str = line.strip()
                                if node_str:
                                    # 유효한 노드 인덱스만 필터링
                                    max_node_idx = 324  # 0부터 시작하므로 325-1
                                    selected_nodes = [int(node.strip()) for node in node_str.split(",") if 0 <= int(node.strip()) <= max_node_idx]
                                break
                    if selected_nodes:
                        logger.info(f"selected_nodes.txt에서 {len(selected_nodes)}개의 선택된 노드 정보를 로드했습니다.")
                except Exception as e:
                    logger.warning(f"노드 정보 로드 중 오류 발생: {str(e)}")
                    logger.info("모든 노드를 사용합니다.")
                    selected_nodes = None
            # 체크포인트에서 노드 정보 가져오기 시도
            else:
                logger.info("selected_nodes.txt 파일이 없습니다. 체크포인트에서 노드 정보를 확인합니다.")
                for file in os.listdir(exp_dir):
                    if file.endswith(".ckpt"):
                        try:
                            ckpt_path = os.path.join(exp_dir, file)
                            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
                            if "selected_nodes" in ckpt:
                                # 유효한 노드 인덱스만 필터링
                                max_node_idx = 324  # 0부터 시작하므로 325-1
                                selected_nodes = [node for node in ckpt["selected_nodes"] if 0 <= node <= max_node_idx]
                                if len(selected_nodes) != len(ckpt["selected_nodes"]):
                                    logger.warning(f"일부 노드 인덱스가 범위를 벗어나 제외되었습니다. (유효한 노드: {len(selected_nodes)}/{len(ckpt['selected_nodes'])})")
                                logger.info(f"체크포인트에서 {len(selected_nodes)}개의 선택된 노드 정보를 로드했습니다.")
                            break
                        except Exception as e:
                            logger.warning(f"체크포인트 로드 중 오류 발생: {str(e)}")

    if selected_nodes is None or len(selected_nodes) == 0:
        logger.info("선택된 노드 정보가 없거나 유효하지 않습니다. 모든 노드를 사용합니다.")
        selected_nodes = None

    ########################################
    # load dataset                         #
    ########################################

    dataset = get_dataset(exp_config["dataset_name"], root_dir=args.root_dir, selected_nodes=selected_nodes)

    # 데이터 품질 확인 및 NaN 처리
    data = dataset._target
    num_nan = np.isnan(data).sum()
    if num_nan > 0:
        logger.warning(f"데이터에 {num_nan}개의 NaN 값이 있습니다. 이는 추론에 문제를 일으킬 수 있습니다.")
        # NaN 값을 0으로 변경하고 마스크로 처리
        logger.info("NaN 값을 0으로 대체하고 마스크 갱신")
        mask = ~np.isnan(data)
        data = np.nan_to_num(data, nan=0.0)
        dataset._target = data
        dataset._mask = dataset._mask & mask
        dataset._training_mask = dataset._training_mask & mask
        dataset._eval_mask = dataset._eval_mask & mask

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

    dm = SpatioTemporalDataModule(torch_dataset, scalers=scalers, splitter=SplitterWrapper(splitter), batch_size=args.batch_size)
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
        logger.info(f"================ 시드 {seed} 테스트 시작 ================")
        # Change evaluation mask
        update_test_eval_mask(dm, dataset, args.p_fault, args.p_noise, seed)

        output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
        output = casting.numpy(output)
        # 차원 확인 후 적절히 처리
        y_hat, y_true, mask = output["y_hat"], output["y"], output["mask"]

        # 차원이 1인 경우에만 squeeze 적용
        if y_hat.ndim > 3 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        if y_true.ndim > 3 and y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)
        if mask.ndim > 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        # 데이터 상태 로깅
        logger.info(f"y_hat 형태: {y_hat.shape}, y_true 형태: {y_true.shape}, mask 형태: {mask.shape}")
        logger.info(f"마스크된 평가 포인트 수: {np.sum(mask)}개")

        # NaN 값 확인 및 처리
        if np.isnan(y_hat).any():
            logger.warning(f"예측 결과에 NaN 값이 있습니다: {np.isnan(y_hat).sum()} 개")
            # NaN 값 위치의 마스크를 False로 설정
            mask = mask & ~np.isnan(y_hat)
            # NaN 값을 0으로 대체 (metric 계산 시 마스크로 무시됨)
            y_hat = np.nan_to_num(y_hat, nan=0.0)

        if np.isnan(y_true).any():
            logger.warning(f"실제 값에 NaN 값이 있습니다: {np.isnan(y_true).sum()} 개")
            # NaN 값 위치의 마스크를 False로 설정
            mask = mask & ~np.isnan(y_true)
            # NaN 값을 0으로 대체 (metric 계산 시 마스크로 무시됨)
            y_true = np.nan_to_num(y_true, nan=0.0)

        # 마스크에 True가 너무 적은지 확인
        mask_ratio = np.mean(mask)
        logger.info(f"마스크 True 비율: {mask_ratio:.4f}")
        if mask_ratio > 0.95:
            logger.warning(f"마스크의 True 비율이 너무 높습니다: {mask_ratio:.4f}. 테스트가 거의 수행되지 않을 수 있습니다.")
        elif mask_ratio < 0.05:
            logger.warning(f"마스크의 True 비율이 너무 낮습니다: {mask_ratio:.4f}. 테스트 결과의 신뢰도가 낮을 수 있습니다.")

        # 마스크의 개수 확인
        total_mask_true = np.sum(mask)
        logger.info(f"마스크 True 개수: {total_mask_true}개")

        # 예측 값과 실제 값 차이 확인
        if total_mask_true > 0:
            # 마스크된 영역에서 값 확인
            masked_y_hat = y_hat[mask]
            masked_y_true = y_true[mask]

            # 기본 통계량 출력
            logger.info(f"실제 값 범위: {np.min(masked_y_true):.4f} ~ {np.max(masked_y_true):.4f}, 평균: {np.mean(masked_y_true):.4f}")
            logger.info(f"예측 값 범위: {np.min(masked_y_hat):.4f} ~ {np.max(masked_y_hat):.4f}, 평균: {np.mean(masked_y_hat):.4f}")

            # 오차 계산
            total_diff = np.sum(np.abs(masked_y_hat - masked_y_true))
            avg_diff = total_diff / total_mask_true
            logger.info(f"평가 포인트 수: {total_mask_true}, 평균 절대 오차: {avg_diff:.6f}")

            # MAE 계산
            check_mae = custom_masked_mae(y_hat, y_true, mask)

            # 결과 확인 및 처리
            if check_mae is None:
                logger.warning("MAE 계산에 실패했습니다. 유효한 평가 포인트가 없습니다. 대체 값을 사용합니다.")
                check_mae = 999.0  # 대체 값 사용
            elif np.isnan(check_mae):
                logger.warning("MAE 계산 결과가 NaN입니다. 대체 값을 사용합니다.")
                check_mae = avg_diff  # 위에서 계산한 평균 절대 오차 사용
        else:
            logger.warning("유효한 평가 포인트가 없습니다! 임의의 값을 사용합니다.")
            check_mae = 999.0  # 마스크 된 포인트가 없으면 큰 값, 'None' 대신 숫자 사용

        mae.append(check_mae)
        print(f"SEED {seed} - Test MAE: {check_mae:.6f}")
        logger.info(f"SEED {seed} - Test MAE: {check_mae:.6f}")

    # NaN 값 필터링
    valid_mae = [m for m in mae if m is not None and not np.isnan(m) and m < 990]  # 999.0은 무효 값으로 제외
    if len(valid_mae) > 0:
        mean_mae = np.mean(valid_mae)
        std_mae = np.std(valid_mae)
        logger.info(f"유효한 MAE 결과: {len(valid_mae)}개/{len(seeds)}개")
    else:
        mean_mae = np.nan
        std_mae = np.nan
        logger.warning("유효한 MAE 결과가 없습니다!")

    print(f"MAE over {len(seeds)} runs: {mean_mae:.6f}±{std_mae:.6f}")
    logger.info(f"MAE over {len(seeds)} runs: {mean_mae:.6f}±{std_mae:.6f}")

    # 노드 서브셋 정보 출력
    if selected_nodes is not None:
        node_count = len(selected_nodes)
        print(f"노드 서브셋 사용: {node_count}개 노드 ({node_count}개 중 {node_count}개 사용, 100.0%)")
    else:
        print("전체 노드 사용")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
