import os

import numpy as np
from tsl.datasets.prototypes import Dataset


# threshold_connectivity 함수가 TSL 라이브러리에 없으므로 직접 구현
def threshold_connectivity(adj_matrix, threshold):
    """임계값 이하의 연결 가중치를 0으로 설정합니다.

    Args:
        adj_matrix: 인접 행렬
        threshold: 임계값

    Returns:
        임계값이 적용된 인접 행렬
    """
    # 임계값보다 작은 값을 0으로 설정
    thresholded = adj_matrix.copy()
    thresholded[thresholded < threshold] = 0
    return thresholded


class STGANBayDataset(Dataset):
    """PemsBay 데이터셋을 STGAN에서 사용하는 형식으로 로드하는 클래스입니다.

    STGAN에서 제공하는 데이터셋 형식:
    - data.npy: 교통 데이터 (시간, 노드, 특성, 채널)
    - time_features.txt: 시간 특성 (요일 7개 + 시간대 24개를 원핫인코딩)
    - node_adjacent.txt: 인접 행렬 (각 노드에 가장 가까운 노드 8개)
    - node_dist.txt: 인접 행렬에 표시된 노드들에 대한 거리 행렬
    """

    def __init__(self, root_dir=None):
        """
        Args:
            root_dir: STGAN 데이터셋이 있는 디렉토리 경로
                기본값은 현재 작업 디렉토리의 '../STGAN/bay'입니다.
        """
        # 먼저 부모 클래스 초기화
        super().__init__()

        # 초기화 후 속성 설정
        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(os.getcwd()), "STGAN", "bay")

        # 클래스 변수로 저장
        self._root_dir = root_dir
        self._data_dir = os.path.join(root_dir, "data")

        # 외부 데이터를 저장할 딕셔너리 초기화
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
            return self._target.shape[2]
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

        # 데이터 형태 변환 (TSL과 호환되도록)
        # STGAN: (time, node, feature, channel) -> TSL: (time, node, channel)
        # STGAN의 feature 차원과 channel 차원을 결합
        n_time, n_nodes, n_features, n_channels = self.data.shape
        self.tsl_data = self.data.reshape(n_time, n_nodes, n_features * n_channels)

        # 인접 행렬 생성
        # STGAN은 각 노드에 대해 가장 가까운 이웃 8개 노드를 저장
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j, neighbor in enumerate(self.node_adjacent[i]):
                if j > 0:  # 첫 번째는 노드 자신일 수 있음
                    adj_matrix[i, neighbor] = 1

        # 인접 행렬의 에지 인덱스 형식으로 직접 변환
        # 인접 행렬에서 0이 아닌 요소 찾기
        rows, cols = np.where(adj_matrix > 0)
        # edge_index 생성 (2 x E 크기, 각 열은 한 에지를 나타냄)
        edge_index = np.array([rows, cols], dtype=np.int64)
        # 가중치 추출
        edge_weight = adj_matrix[rows, cols]

        # 속성 직접 설정
        self._target = self.tsl_data
        self._connectivity = adj_matrix
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self._mask = np.ones_like(self.tsl_data, dtype=bool)
        self._training_mask = self._mask.copy()
        self._eval_mask = self._mask.copy()

        # 시간 정보 추가
        self.add_exogenous("temporal_encoding", self.time_features)

        # Dataset 클래스의 필수 속성 설정
        self._idx = np.arange(self.tsl_data.shape[0])
        self._t = np.arange(self.tsl_data.shape[0])

        return self

    def add_exogenous(self, name, data, axis=0):
        """외부 데이터 추가

        Args:
            name: 외부 데이터 이름
            data: 추가할 데이터
            axis: 정렬 축
        """
        # 데이터가 리스트 형태면 배열로 변환
        if isinstance(data, list):
            data = np.array(data)

        # 데이터가 0 축이 데이터의 길이와 일치하는지 확인
        if axis == 0 and len(data) != self._target.shape[0]:
            raise ValueError(f"데이터 길이 불일치: {len(data)} vs {self._target.shape[0]}")

        # 이름과 데이터 저장
        self._exogenous[name] = (data, axis)

        return self

    def get_exogenous(self, name):
        """저장된 외부 데이터 반환

        Args:
            name: 외부 데이터 이름

        Returns:
            저장된 외부 데이터
        """
        if name not in self._exogenous:
            raise KeyError(f"외부 데이터 '{name}'이(가) 없습니다.")
        return self._exogenous[name][0]

    def set_eval_mask(self, mask):
        """평가 마스크 설정

        Args:
            mask: 평가 마스크
        """
        self._eval_mask = mask.copy()

    def datetime_encoded(self, fields=None):
        """요일 및 시간을 인코딩한 특성 반환

        Args:
            fields: 사용할 필드 ('day', 'week')

        Returns:
            인코딩된 시간 특성
        """
        # STGAN의 time_features는 이미 인코딩된 형태로 제공됨
        # 앞 7개 열은 요일, 뒤 24개 열은 시간대
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
        """연결 행렬 반환

        Args:
            threshold: 연결 임계값
            include_self: 자기 연결 포함 여부
            force_symmetric: 대칭 행렬 강제

        Returns:
            인접 행렬
        """
        # STGAN에서 제공하는 인접 정보로부터 연결 행렬 생성
        n_nodes = self.node_adjacent.shape[0]
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j, neighbor in enumerate(self.node_adjacent[i]):
                if j > 0:  # 첫 번째는 노드 자신일 수 있음
                    # 거리에 따른 가중치 적용
                    adj_matrix[i, neighbor] = 1 / (self.node_dist[i, j] + 1e-10)

        # 대칭 행렬로 만들기
        if force_symmetric:
            adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

        # 자기 연결 추가
        if include_self:
            np.fill_diagonal(adj_matrix, 1.0)

        # 임계값 적용
        if threshold is not None:
            adj_matrix = threshold_connectivity(adj_matrix, threshold)

        return adj_matrix

    # Dataset 클래스의 필수 메서드 구현
    def numpy(self, return_idx=False):
        if return_idx:
            return self._target, self._idx
        return self._target

    def get_splitter(self, method="temporal", **kwargs):
        """데이터셋 분할기 반환

        Args:
            method: 분할 방법 (무시됨, 항상 시간 기반 분할 사용)
            **kwargs: 추가 인수, val_len과 test_len이 포함될 수 있음

        Returns:
            분할 인덱스 함수
        """
        # 추가 인수에서 검증/테스트 세트 비율 가져오기
        val_len = kwargs.get("val_len", 0.1)
        test_len = kwargs.get("test_len", 0.2)

        # 전체 길이
        total_len = len(self._idx)

        # 인덱스 계산
        train_end = int(total_len * (1 - val_len - test_len))
        val_end = int(total_len * (1 - test_len))

        train_idx = self._idx[:train_end]
        val_idx = self._idx[train_end:val_end]
        test_idx = self._idx[val_end:]

        # 분할 함수 정의
        def splitter(idx, **kwargs):
            # 훈련/검증/테스트 세트 인덱스 반환
            train_mask = np.isin(idx, train_idx)
            val_mask = np.isin(idx, val_idx)
            test_mask = np.isin(idx, test_idx)

            return {"train": train_mask, "val": val_mask, "test": test_mask}

        return splitter
