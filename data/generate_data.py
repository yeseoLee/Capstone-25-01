import os
import pickle
import urllib.request
import zipfile

import numpy as np
import torch
from torch_geometric.data import Data


class TrafficDataLoader:
    def __init__(self, dataset="PEMS-BAY", data_dir="data/raw"):
        """
        교통 데이터 로더
        Args:
            dataset: 'PEMS-BAY' 또는 'METR-LA'
            data_dir: 데이터 저장 경로
        """
        self.dataset = dataset
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.url_dict = {
            "PEMS-BAY": "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip",
            "METR-LA": "https://graphmining.ai/temporal_datasets/METR-LA.zip",
        }

    def download_and_extract(self):
        """데이터셋 다운로드 및 압축 해제"""
        if self.dataset not in self.url_dict:
            raise ValueError(f"지원하지 않는 데이터셋: {self.dataset}. 'PEMS-BAY' 또는 'METR-LA'를 사용하세요.")

        data_path = os.path.join(self.data_dir, f"{self.dataset}.zip")
        if not os.path.exists(data_path):
            print(f"{self.dataset} 데이터셋 다운로드 중...")
            urllib.request.urlretrieve(self.url_dict[self.dataset], data_path)

        extract_path = os.path.join(self.data_dir, self.dataset)
        if not os.path.exists(extract_path):
            print(f"{self.dataset} 압축 해제 중...")
            with zipfile.ZipFile(data_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

        return extract_path

    def load_data(self):
        """데이터 로드"""
        extract_path = self.download_and_extract()

        # 데이터 로드
        data_path = os.path.join(extract_path, f"{self.dataset}.npz")
        data = np.load(data_path)

        # 속성 정보
        attributes = {}
        for key in data.files:
            attributes[key] = data[key]

        adj_mx = attributes["adj_mx"]
        node_values = attributes["node_values"]  # (num_nodes, num_timesteps, num_features)

        print(f"데이터 형태: {node_values.shape}")
        print(f"인접 행렬 형태: {adj_mx.shape}")

        return node_values, adj_mx

    def create_pyg_data(self, node_values, adj_mx):
        """PyTorch Geometric 데이터 객체 생성"""
        # 엣지 리스트 생성
        edges = np.nonzero(adj_mx)
        edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)

        # 노드 특성 (시간 차원은 유지)
        x = torch.tensor(node_values, dtype=torch.float)

        # PyG 데이터 객체 생성
        data = Data(x=x, edge_index=edge_index)

        return data


def generate_missing_values(data, missing_rate=0.2, random_seed=42):
    """
    결측치 생성
    Args:
        data: 원본 데이터 (numpy 배열)
        missing_rate: 결측치 비율 (0~1)
        random_seed: 랜덤 시드

    Returns:
        missing_mask: 결측치 마스크 (True: 결측, False: 관측)
        data_with_missing: 결측치가 포함된 데이터 (결측치는 np.nan)
    """
    np.random.seed(random_seed)

    # 데이터 복사
    data_with_missing = data.copy()

    # 결측치 마스크 생성
    missing_mask = np.random.rand(*data.shape) < missing_rate

    # 결측치 적용
    data_with_missing[missing_mask] = np.nan

    return missing_mask, data_with_missing


def generate_outliers(data, outlier_rate=0.1, std_multiplier=3, random_seed=42):
    """
    이상치 생성
    Args:
        data: 원본 데이터 (numpy 배열)
        outlier_rate: 이상치 비율 (0~1)
        std_multiplier: 표준편차 배수 (이상치 크기 결정)
        random_seed: 랜덤 시드

    Returns:
        outlier_mask: 이상치 마스크 (True: 이상치, False: 정상치)
        data_with_outliers: 이상치가 포함된 데이터
    """
    np.random.seed(random_seed)

    # 데이터 복사
    data_with_outliers = data.copy()

    # 각 노드별, 특성별 통계 계산
    data_mean = np.nanmean(data, axis=1, keepdims=True)
    data_std = np.nanstd(data, axis=1, keepdims=True)

    # 이상치 마스크 생성
    outlier_mask = np.random.rand(*data.shape) < outlier_rate

    # 이상치 생성 (평균에서 표준편차의 몇 배 떨어진 값)
    outlier_direction = np.random.choice([-1, 1], size=outlier_mask.sum())

    # 이상치 적용
    outlier_values = (
        data_mean.repeat(data.shape[1], axis=1)[outlier_mask]
        + outlier_direction * std_multiplier * data_std.repeat(data.shape[1], axis=1)[outlier_mask]
    )
    data_with_outliers[outlier_mask] = outlier_values

    return outlier_mask, data_with_outliers


def generate_mixed_data(data, missing_rate=0.2, outlier_rate=0.1, std_multiplier=3, random_seed=42):
    """
    결측치와 이상치가 혼합된 데이터 생성
    Args:
        data: 원본 데이터 (numpy 배열)
        missing_rate: 결측치 비율 (0~1)
        outlier_rate: 이상치 비율 (0~1)
        std_multiplier: 표준편차 배수 (이상치 크기 결정)
        random_seed: 랜덤 시드

    Returns:
        missing_mask: 결측치 마스크
        outlier_mask: 이상치 마스크
        mixed_data: 결측치와 이상치가 혼합된 데이터
    """
    np.random.seed(random_seed)

    # 이상치 생성
    outlier_mask, data_with_outliers = generate_outliers(data, outlier_rate, std_multiplier, random_seed)

    # 결측치 생성 (이상치 마스크와 겹치지 않도록)
    missing_candidates = ~outlier_mask
    missing_prob = np.random.rand(*data.shape) * missing_candidates
    # 전체 대비 missing_rate 비율만큼 결측치 생성
    missing_threshold = np.percentile(missing_prob[missing_candidates], (1 - missing_rate) * 100)
    missing_mask = missing_prob > missing_threshold

    # 결측치 적용
    mixed_data = data_with_outliers.copy()
    mixed_data[missing_mask] = np.nan

    return missing_mask, outlier_mask, mixed_data


def save_dataset(data_dict, filename):
    """데이터셋 저장"""
    with open(filename, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"데이터셋 저장 완료: {filename}")


def load_dataset(filename):
    """데이터셋 로드"""
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)
    print(f"데이터셋 로드 완료: {filename}")
    return data_dict


def main():
    """메인 함수"""
    # 데이터 로더 초기화
    loader = TrafficDataLoader(dataset="PEMS-BAY")

    # 데이터 로드
    node_values, adj_mx = loader.load_data()

    # 원본 데이터 준비 (차원 변경: (노드, 시간, 특성) -> (노드, 시간))
    original_data = node_values[:, :, 0]  # 첫 번째 특성만 사용

    # 결측치 및 이상치 생성
    missing_mask, outlier_mask, mixed_data = generate_mixed_data(original_data, missing_rate=0.2, outlier_rate=0.1)

    # 데이터셋 저장
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    data_dict = {
        "original_data": original_data,
        "mixed_data": mixed_data,
        "missing_mask": missing_mask,
        "outlier_mask": outlier_mask,
        "adj_mx": adj_mx,
    }

    save_dataset(data_dict, os.path.join(processed_dir, "traffic_data.pkl"))

    # PyG 데이터 객체 생성 및 저장
    # PyG 데이터는 나중에 모델에서 필요할 때 로드하고 변환


if __name__ == "__main__":
    main()
