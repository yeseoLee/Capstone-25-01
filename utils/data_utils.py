import pickle

import numpy as np
import torch
from torch_geometric.data import Data


def load_dataset(filename):
    """
    데이터셋 로드
    Args:
        filename: 데이터셋 파일 경로
    """
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def create_torch_dataset(data_dict, window_size=12, stride=1):
    """
    시계열 윈도우 데이터셋 생성
    Args:
        data_dict: 데이터 딕셔너리
        window_size: 윈도우 크기
        stride: 윈도우 간 스트라이드

    Returns:
        dataset: (입력 윈도우, 타겟 윈도우) 튜플 리스트
    """
    original_data = data_dict["original_data"]
    mixed_data = data_dict["mixed_data"]
    adj_mx = data_dict["adj_mx"]
    missing_mask = data_dict["missing_mask"]
    outlier_mask = data_dict["outlier_mask"]

    n_nodes, n_timestamps = original_data.shape

    # 윈도우 인덱스 생성
    indices = [(i, i + window_size) for i in range(0, n_timestamps - window_size + 1, stride)]

    dataset = []

    for start_idx, end_idx in indices:
        # 각 윈도우의 데이터 추출
        window_original = original_data[:, start_idx:end_idx]
        window_mixed = mixed_data[:, start_idx:end_idx]
        window_missing_mask = missing_mask[:, start_idx:end_idx]
        window_outlier_mask = outlier_mask[:, start_idx:end_idx]

        # PyTorch 텐서로 변환
        x = torch.tensor(window_mixed, dtype=torch.float32)
        y = torch.tensor(window_original, dtype=torch.float32)
        missing_mask_tensor = torch.tensor(window_missing_mask, dtype=torch.bool)
        outlier_mask_tensor = torch.tensor(window_outlier_mask, dtype=torch.bool)

        # 엣지 인덱스 생성
        edge_index = torch.tensor(np.array(np.nonzero(adj_mx)), dtype=torch.long)

        # PyG 데이터 객체 생성
        data = Data(
            x=x,  # 입력 (손상된 데이터)
            y=y,  # 타겟 (원본 데이터)
            edge_index=edge_index,  # 엣지 인덱스
            missing_mask=missing_mask_tensor,  # 결측치 마스크
            outlier_mask=outlier_mask_tensor,  # 이상치 마스크
        )

        dataset.append(data)

    return dataset


def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    데이터로더 생성
    Args:
        dataset: 데이터셋
        batch_size: 배치 크기
        shuffle: 셔플 여부
    """
    from torch_geometric.loader import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def normalize_data(data, scaler=None, axis=1):
    """
    데이터 정규화
    Args:
        data: 정규화할 데이터
        scaler: 스케일러 (None일 경우 새로 생성)
        axis: 정규화 축 (0: 노드별, 1: 시간별)

    Returns:
        normalized_data: 정규화된 데이터
        scaler: 사용된 스케일러 (mean, std)
    """
    if scaler is None:
        # NaN 값을 제외하고 평균과 표준편차 계산
        mean = np.nanmean(data, axis=axis, keepdims=True)
        std = np.nanstd(data, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # 0으로 나누기 방지
        scaler = (mean, std)
    else:
        mean, std = scaler

    # 정규화
    normalized_data = (data - mean) / std

    return normalized_data, scaler


def denormalize_data(data, scaler):
    """
    정규화된 데이터 원복
    Args:
        data: 정규화된 데이터
        scaler: 정규화에 사용된 스케일러 (mean, std)

    Returns:
        denormalized_data: 원복된 데이터
    """
    mean, std = scaler
    return data * std + mean


def split_data(data_dict, train_ratio=0.7, val_ratio=0.1):
    """
    데이터 분할 (학습, 검증, 테스트)
    Args:
        data_dict: 데이터 딕셔너리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율 (나머지는 테스트 데이터)

    Returns:
        train_data: 학습 데이터 딕셔너리
        val_data: 검증 데이터 딕셔너리
        test_data: 테스트 데이터 딕셔너리
    """
    n_timestamps = data_dict["original_data"].shape[1]

    # 시간 인덱스 분할
    train_end = int(n_timestamps * train_ratio)
    val_end = train_end + int(n_timestamps * val_ratio)

    # 딕셔너리 복사
    train_data = {}
    val_data = {}
    test_data = {}

    # 각 데이터 아이템 분할
    for key, value in data_dict.items():
        if key == "adj_mx":  # 인접 행렬은 분할하지 않음
            train_data[key] = val_data[key] = test_data[key] = value
        else:  # 시계열 데이터 분할
            train_data[key] = value[:, :train_end]
            val_data[key] = value[:, train_end:val_end]
            test_data[key] = value[:, val_end:]

    return train_data, val_data, test_data


def mask_to_index(mask):
    """
    마스크를 인덱스로 변환
    Args:
        mask: Boolean 마스크

    Returns:
        indices: 마스크 True 위치의 인덱스 (2D)
    """
    return np.array(np.where(mask)).T
