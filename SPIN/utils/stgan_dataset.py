import os

import numpy as np
from tsl.datasets.prototypes import Dataset
from tsl.ops.connectivity import adj_to_edge_index, threshold_connectivity


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
                기본값은 현재 작업 디렉토리의 'STGAN/bay'입니다.
        """
        if root_dir is None:
            root_dir = os.path.join(os.getcwd(), "STGAN", "bay")

        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")

        super().__init__()

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

        # 인접 행렬의 에지 인덱스 형식으로 변환
        edge_index, edge_weight = adj_to_edge_index(adj_matrix)

        # 데이터셋 정보 설정
        super().load(target=self.tsl_data, connectivity=adj_matrix, edge_index=edge_index, edge_weight=edge_weight)

        # 시간 정보 추가
        self.add_exogenous("temporal_encoding", self.time_features)

        return self

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
