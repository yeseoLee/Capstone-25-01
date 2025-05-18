import os

import numpy as np
import torch
import torch.utils.data as data


class STGANDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # 데이터 파일 경로
        data_path = os.path.join(data_dir, "data.npy")  # traffic data
        feature_path = os.path.join(data_dir, "time_features.txt")  # time feature
        graph_path = os.path.join(data_dir, "node_subgraph.npy")  # (num_node, n, n), the subgraph of each node
        adj_path = os.path.join(data_dir, "node_adjacent.txt")  # (num_node, n), the adjacent of each node

        # 데이터 로드
        self.data = torch.tensor(np.load(data_path), dtype=torch.float)
        self.time_features = torch.tensor(np.loadtxt(feature_path), dtype=torch.float)
        self.graph = torch.tensor(np.load(graph_path), dtype=torch.float)
        self.adjs = torch.tensor(np.loadtxt(adj_path), dtype=torch.int)
        print(f"데이터 로드 완료: {self.data.shape}")

        # 차원 및 크기 정보
        self.n_nodes = self.data.shape[1]  # 노드 수
        self.n_features = self.data.shape[2]  # 특성 수
        self.n_channels = self.data.shape[3]  # 채널 수
        self.seq_len = self.data.shape[0]  # 시퀀스 길이

        # 인접 노드 수
        self.n_adj = self.adjs.shape[1]

        # 이동 평균 계산을 위한 윈도우 크기
        self.window_size = 12  # 1시간 (5분 간격 데이터)

        # normalization
        self._normalize_data()
        self._weight_graph()

        # 유효한 데이터 인덱스 계산 (첫 window_size 개는 제외)
        self.valid_indices = list(range(self.window_size, self.seq_len))
        self.length = len(self.valid_indices) * self.n_nodes

    def _normalize_data(self):
        """데이터 정규화: Min-Max 스케일링 적용"""
        # 채널별 정규화
        for c in range(self.n_channels):
            max_val = torch.max(self.data[:, :, :, c])
            min_val = torch.min(self.data[:, :, :, c])
            self.data[:, :, :, c] = self._min_max_scale(self.data[:, :, :, c], max_val, min_val)

    def _min_max_scale(self, data, max_val, min_val):
        """Min-Max 스케일링: [-1, 1] 범위로 변환"""
        normalized = (data - min_val) / (max_val - min_val + 1e-5)
        return normalized * 2 - 1

    def _weight_graph(self):
        """그래프 가중치 계산"""
        # 노드 간 거리 정보 로드
        dist_path = os.path.join(self.data_dir, "node_dist.txt")
        if os.path.exists(dist_path):
            distances = torch.tensor(np.loadtxt(dist_path), dtype=torch.float)
            # 거리 기반 가우시안 가중치 계산
            sigma = torch.std(distances)
            self.graph = torch.exp(-torch.pow(self.graph, 2) / (2 * torch.pow(sigma, 2)))

    def calculate_normalized_laplacian(self, adj):
        """정규화된 라플라시안 행렬 계산
        L = D^(-1/2) * (D - A) * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
        """
        # 자기 루프 추가
        adj = adj + torch.eye(adj.shape[0], device=adj.device)

        # 차수 행렬의 역제곱근 계산
        d_inv_sqrt = torch.pow(torch.sum(adj, dim=1) + 1e-5, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        # 정규화된 라플라시안 계산
        norm_lap = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        return norm_lap

    def __getitem__(self, idx):
        """데이터셋에서 단일 샘플 가져오기"""
        # 시간 및 노드 인덱스 계산
        node_idx = idx % self.n_nodes
        time_idx = self.valid_indices[idx // self.n_nodes]

        # 최근 시계열 데이터 준비 (window_size 기간)
        recent_data = torch.zeros((self.window_size, self.n_adj, self.n_features * self.n_channels))
        real_data = torch.zeros((self.n_adj, self.n_features * self.n_channels))

        # 인접 노드에 대한 데이터 구성
        for i in range(self.n_adj):
            adj_node = self.adjs[node_idx, i]
            # 최근 데이터 (과거 window_size 기간)
            recent_data[:, i, :] = self.data[time_idx - self.window_size : time_idx, adj_node, :, :].reshape(
                self.window_size, -1
            )
            # 현재 데이터 (예측 대상)
            real_data[i, :] = self.data[time_idx, adj_node, :, :].reshape(-1)

        # 추세 데이터 (같은 노드의 과거 데이터)
        trend_len = self.window_size * 7  # 7일 데이터 (주간 패턴)
        t_start = max(0, time_idx - trend_len)
        trend_data = self.data[t_start:time_idx, node_idx, :, :].reshape(time_idx - t_start, -1)

        # 시간 특성 추출
        time_feature = self.time_features[time_idx]

        # 서브그래프 행렬 추출 및 정규화
        subgraph = self.graph[node_idx].clone()
        subgraph = self.calculate_normalized_laplacian(subgraph)

        # input_data: 최근 데이터, 추세 데이터, 시간 특성
        # target: 실제 데이터
        # 각 샘플은 (input_data, subgraph, target) 형태로 반환
        return (recent_data, trend_data, time_feature), subgraph, real_data

    def __len__(self):
        """데이터셋 크기 반환"""
        return self.length
