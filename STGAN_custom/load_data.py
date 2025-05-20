import logging
import os

import numpy as np
import torch
import torch.utils.data as data


class data_loader(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        # 로거 설정
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("STGAN-DataLoader")

        # 경로 설정 수정 - 이중 슬래시 문제 해결
        data_path = os.path.join(opt["data_path"], "data.npy")  # traffic data
        feature_path = os.path.join(opt["data_path"], "time_features.txt")  # time feature
        graph_path = os.path.join(opt["data_path"], "node_subgraph.npy")  # (num_node, n, n), the subgraph of each node
        adj_path = os.path.join(opt["data_path"], "node_adjacent.txt")  # (num_node, n), the adjacent of each node

        # 파일 존재 확인
        for file_path, file_desc in [(data_path, "데이터 파일"), (feature_path, "시간 특성 파일"), (graph_path, "노드 서브그래프 파일"), (adj_path, "노드 인접 파일")]:
            if not os.path.exists(file_path):
                self.logger.error(f"{file_desc}이(가) 존재하지 않습니다: {file_path}")
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            else:
                self.logger.info(f"{file_desc} 확인: {file_path}")

        # 원본 데이터 로드
        original_data = np.load(data_path)
        self.time_features = torch.tensor(np.loadtxt(feature_path), dtype=torch.float)
        original_graph = np.load(graph_path)
        original_adjs = np.loadtxt(adj_path).astype(int)  # int로 명시적 변환

        # 원본 노드 수 저장
        self.original_node_num = original_data.shape[1]
        self.logger.info(f"원본 데이터 형태: {original_data.shape}, 원본 노드 수: {self.original_node_num}")

        # 노드 서브셋 처리
        self.selected_nodes = None
        self.node_index_map = None  # 원본 노드 인덱스 -> 새 인덱스 매핑

        if "use_node_subset" in opt and opt["use_node_subset"]:
            # 노드 리스트가 직접 지정된 경우
            if "node_list" in opt and opt["node_list"]:
                self.selected_nodes = [int(node) for node in opt["node_list"].split(",")]
                self.logger.info(f"사용자 지정 {len(self.selected_nodes)}개 노드를 사용합니다: {self.selected_nodes}")
            else:
                # 비율에 따라 랜덤하게 노드 선택
                np.random.seed(opt["seed"])  # 동일한 시드 사용
                num_nodes = max(1, int(self.original_node_num * opt["node_ratio"]))
                self.selected_nodes = np.random.choice(self.original_node_num, num_nodes, replace=False).tolist()
                self.logger.info(f"전체 {self.original_node_num}개 노드 중 {num_nodes}개({opt['node_ratio']*100:.1f}%)를 랜덤하게 선택했습니다.")

            # 선택된 노드 정보 저장
            if "node_info_file" in opt:
                with open(opt["node_info_file"], "w") as f:
                    f.write(f"# 총 {len(self.selected_nodes)}개의 노드\n")
                    f.write(", ".join(map(str, self.selected_nodes)))
                self.logger.info(f"선택된 노드 정보를 {opt['node_info_file']}에 저장했습니다.")

            # 원본 노드 인덱스를 새 인덱스로 매핑하는 사전 생성
            self.node_index_map = {node_idx: i for i, node_idx in enumerate(self.selected_nodes)}

            # 인접 노드 리스트 조정
            # 인접 노드가 선택된 노드에 포함되지 않는 경우, 가장 가까운 선택된 노드로 대체
            # 1. 원본 노드만 포함하는 새 인접 행렬 생성
            new_adjs = np.zeros((len(self.selected_nodes), original_adjs.shape[1]), dtype=int)

            for i, node_idx in enumerate(self.selected_nodes):
                for j in range(original_adjs.shape[1]):
                    adj_node = original_adjs[node_idx, j]

                    # 인접 노드가 선택된 노드 목록에 있는지 확인
                    if adj_node in self.selected_nodes:
                        # 선택된 노드 목록 내에서의 인덱스로 변경
                        new_adjs[i, j] = self.node_index_map[adj_node]
                    else:
                        # 인접 노드가 선택되지 않은 경우, 자기 자신으로 대체
                        # 이 부분은 데이터셋의 특성에 따라 다르게 처리할 수 있음
                        new_adjs[i, j] = i  # 자기 자신으로 대체

            self.logger.info("인접 노드 정보를 노드 서브셋에 맞게 조정했습니다.")

            # 선택된 노드만 사용하도록 데이터 필터링
            self.data = torch.tensor(original_data[:, self.selected_nodes, :, :], dtype=torch.float)
            self.graph = torch.tensor(original_graph[self.selected_nodes], dtype=torch.float)
            self.adjs = torch.tensor(new_adjs, dtype=torch.long)  # long으로 변경
            self.logger.info(f"노드 서브셋 적용 후 데이터 형태: {self.data.shape}")
            self.logger.info(f"조정된 인접 노드 행렬 형태: {self.adjs.shape}")
        else:
            # 모든 노드 사용
            self.data = torch.tensor(original_data, dtype=torch.float)
            self.graph = torch.tensor(original_graph, dtype=torch.float)
            self.adjs = torch.tensor(original_adjs, dtype=torch.long)  # long으로 변경
            self.logger.info("전체 노드를 사용합니다.")

        self.logger.info(f"traffic data: {self.data.shape}")

        # direction subgraph, no self connect
        self.T_recent = opt["recent_time"] * opt["timestamp"]
        self.T_trend = opt["trend_time"] * opt["timestamp"]
        if opt["isTrain"]:
            self.start_time = self.T_trend
            self.time_num = opt["train_time"] - self.start_time
        else:
            self.start_time = opt["train_time"]
            self.time_num = self.data.shape[0] - self.start_time

        self.input_size = self.data.shape[2] * self.data.shape[3]

        self.adj_num = self.adjs.shape[1]
        self.node_num = self.data.shape[1]

        # normalize
        self.normalize()
        self.weight()

        self.length = self.node_num * self.time_num

    def __getitem__(self, idx):
        index_t = idx // self.node_num + self.start_time
        index_r = idx % self.node_num

        # recent_data: (time, sub_graph, num_feature)
        recent_data = torch.zeros((self.T_recent, self.adj_num, self.input_size))
        real_data = torch.zeros((self.adj_num, self.input_size))

        # 인덱스 범위 안전 확인
        if index_t < self.T_recent or index_t >= self.data.shape[0]:
            self.logger.error(f"시간 인덱스가 범위를 벗어났습니다: {index_t}, 범위: [0, {self.data.shape[0]-1}]")
            # 안전한 인덱스 사용
            index_t = max(self.T_recent, min(index_t, self.data.shape[0] - 1))

        # recent
        for i in range(self.adj_num):
            # 인접 노드 인덱스 안전 확인
            adj_idx = self.adjs[index_r, i].item()
            if adj_idx < 0 or adj_idx >= self.node_num:
                self.logger.warning(f"인접 노드 인덱스가 범위를 벗어났습니다: {adj_idx}, 범위: [0, {self.node_num-1}]")
                adj_idx = max(0, min(adj_idx, self.node_num - 1))

            recent_data[:, i, :] = self.data[index_t - self.T_recent : index_t, adj_idx, :, :].view(self.T_recent, -1)
            real_data[i, :] = self.data[index_t, adj_idx, :, :].view(-1)

        # trend
        trend_data = self.data[index_t - self.T_trend : index_t, index_r, :].view(self.T_trend, -1)
        time_feature = self.time_features[index_t,]
        subgraph = self.graph[index_r,]
        subgraph = self.calculate_normalized_laplacian(subgraph)

        return (recent_data, trend_data, time_feature), subgraph, real_data, index_t - self.start_time, index_r

    def weight(self):
        # std
        # 경로 설정 수정
        dist_path = os.path.join(self.opt["data_path"], "node_dist.txt")

        if not os.path.exists(dist_path):
            self.logger.error(f"노드 거리 파일이 존재하지 않습니다: {dist_path}")
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {dist_path}")

        dists = torch.tensor(np.loadtxt(dist_path), dtype=torch.float)
        # 노드 서브셋 사용 시 해당 노드의 거리만 사용
        if self.selected_nodes is not None:
            dists = dists[self.selected_nodes]
        delta = torch.std(dists)
        self.graph = torch.exp(-np.divide(np.power(self.graph, 2), np.power(delta, 2)))

    def calculate_normalized_laplacian(self, adj):
        """
        # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
        # D = diag(A 1)
        :param adj:
        :return:
        """
        # A = A + I
        adj += torch.eye(adj.shape[0])
        d_inv_sqrt = (torch.sum(adj, 1) + 1e-5) ** (-0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        normalized_laplacian = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        return normalized_laplacian

    def normalize(self):
        # 노드 서브셋 사용 시 학습 데이터의 범위를 올바르게 계산해야 함
        if self.opt["isTrain"]:
            max_source1 = torch.max(self.data[: self.opt["train_time"], :, :, 0])
            min_source1 = torch.min(self.data[: self.opt["train_time"], :, :, 0])
            max_source2 = torch.max(self.data[: self.opt["train_time"], :, :, 1])
            min_source2 = torch.min(self.data[: self.opt["train_time"], :, :, 1])

            # 최대/최소값을 로그에 출력
            self.logger.info(f"데이터 정규화 범위 - Source1: [{min_source1.item()}, {max_source1.item()}], Source2: [{min_source2.item()}, {max_source2.item()}]")

            # 정규화 범위가 0인 경우(모든 값이 같은 경우) 경고 출력
            if max_source1 == min_source1:
                self.logger.warning(f"Source1의 정규화 범위가 0입니다! 모든 값: {max_source1.item()}")
            if max_source2 == min_source2:
                self.logger.warning(f"Source2의 정규화 범위가 0입니다! 모든 값: {max_source2.item()}")

            # 노드 서브셋 사용 시 정규화 범위를 저장
            if self.selected_nodes is not None and "node_info_file" in self.opt:
                normalization_info_file = self.opt["node_info_file"].replace("selected_nodes", "normalization_info")
                with open(normalization_info_file, "w") as f:
                    f.write(f"Source1 정규화 범위: [{min_source1.item()}, {max_source1.item()}]\n")
                    f.write(f"Source2 정규화 범위: [{min_source2.item()}, {max_source2.item()}]\n")
                self.logger.info(f"정규화 정보를 {normalization_info_file}에 저장했습니다.")

            self.data[:, :, :, 0] = self.max_min(self.data[:, :, :, 0], max_source1, min_source1)
            self.data[:, :, :, 1] = self.max_min(self.data[:, :, :, 1], max_source2, min_source2)
        else:
            # 테스트 모드에서는 이전에 저장된 정규화 범위를 로드하여 사용할 수도 있음
            self.logger.info("테스트 모드에서 데이터 정규화 수행")
            max_source1 = torch.max(self.data[: self.opt["train_time"], :, :, 0])
            min_source1 = torch.min(self.data[: self.opt["train_time"], :, :, 0])
            max_source2 = torch.max(self.data[: self.opt["train_time"], :, :, 1])
            min_source2 = torch.min(self.data[: self.opt["train_time"], :, :, 1])

            self.data[:, :, :, 0] = self.max_min(self.data[:, :, :, 0], max_source1, min_source1)
            self.data[:, :, :, 1] = self.max_min(self.data[:, :, :, 1], max_source2, min_source2)

    def max_min(self, data, max_val, min_val):
        # 정규화 범위가 0인 경우(모든 값이 같은 경우) 나눗셈 오류 방지
        if max_val == min_val:
            self.logger.warning("정규화 범위가 0입니다! 원본 데이터를 유지합니다.")
            return torch.zeros_like(data)

        data = (data - min_val) / (max_val - min_val)
        data = data * 2 - 1

        return data

    def __len__(self):
        return self.length

    def get_selected_nodes(self):
        """선택된 노드 리스트를 반환합니다. 전체 노드 사용 시 None을 반환합니다."""
        return self.selected_nodes
