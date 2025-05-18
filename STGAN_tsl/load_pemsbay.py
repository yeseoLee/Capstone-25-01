"""
TSL PemsBay 데이터셋 로드 모듈

이 모듈은 TSL(Torch Spatiotemporal Library)의 PemsBay 데이터셋을 로드하고
전처리하는 기능을 제공합니다.

기본 사용법:
    python load_pemsbay.py
"""

import os

import numpy as np
from tsl.datasets import PemsBay
from tsl.ops.imputation import add_missing_values


def load_pemsbay_dataset(p_fault=0.0015, p_noise=0.05, seed=42):
    """
    PemsBay 데이터셋을 로드하고 결측치를 추가합니다.

    Args:
        p_fault: 연속적인 결측치를 생성하는 비율 (기본값: 0.0015)
        p_noise: 독립적인 결측치를 생성하는 비율 (기본값: 0.05)
        seed: 랜덤 시드 (기본값: 42)

    Returns:
        dataset: 결측치가 추가된 PemsBay 데이터셋
    """
    # 원본 PemsBay 데이터셋 로드
    print("PemsBay 데이터셋 로드 중...")
    dataset = PemsBay()

    # 데이터셋 기본 정보 출력
    print(f"데이터셋 크기: {dataset.shape}")
    print(f"노드 수: {dataset.n_nodes}")
    print(f"시간 간격: {dataset.freq}")

    # 결측치 추가
    print(f"결측치 추가 중... (p_fault={p_fault}, p_noise={p_noise}, seed={seed})")
    dataset_with_missing = add_missing_values(
        dataset,
        p_fault=p_fault,
        p_noise=p_noise,
        min_seq=12,  # 최소 연속 결측 길이 (블록 결측치 용)
        max_seq=12 * 4,  # 최대 연속 결측 길이 (블록 결측치 용)
        seed=seed,
    )

    # 결측치 통계 계산
    missing_ratio = 1.0 - np.mean(dataset_with_missing.training_mask)
    print(f"추가된 결측치 비율: {missing_ratio:.4f} ({missing_ratio*100:.2f}%)")

    return dataset_with_missing


def save_pemsbay_data(dataset, output_dir="./bay/data"):
    """
    PemsBay 데이터셋을 NumPy 형식으로 저장합니다.

    Args:
        dataset: 저장할 PemsBay 데이터셋
        output_dir: 저장할 디렉토리 경로 (기본값: "./bay/data")
    """
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 추출
    data, mask = dataset.numpy(return_mask=True)

    # 데이터 저장
    np.save(os.path.join(output_dir, "pemsbay_data.npy"), data)
    np.save(os.path.join(output_dir, "pemsbay_mask.npy"), mask)

    # 인접 행렬 저장
    adj = dataset.get_connectivity(threshold=0.1, include_self=False)
    np.save(os.path.join(output_dir, "pemsbay_adj.npy"), adj)

    # 시간 정보 저장
    time_encoded = dataset.datetime_encoded(["day", "week"]).values
    np.save(os.path.join(output_dir, "pemsbay_time_features.npy"), time_encoded)

    print(f"데이터가 {output_dir} 디렉토리에 저장되었습니다.")
    print("파일 목록:")
    print(f"- pemsbay_data.npy: 데이터 배열 (크기: {data.shape})")
    print(f"- pemsbay_mask.npy: 마스크 배열 (크기: {mask.shape})")
    print(f"- pemsbay_adj.npy: 인접 행렬 (크기: {adj.shape})")
    print(f"- pemsbay_time_features.npy: 시간 특성 (크기: {time_encoded.shape})")


def create_point_missing(p_noise=0.25, seed=42):
    """
    포인트 결측치가 있는 PemsBay 데이터셋을 생성합니다.

    Args:
        p_noise: 독립적인 결측치를 생성하는 비율 (기본값: 0.25)
        seed: 랜덤 시드 (기본값: 42)

    Returns:
        dataset: 포인트 결측치가 추가된 PemsBay 데이터셋
    """
    return load_pemsbay_dataset(p_fault=0.0, p_noise=p_noise, seed=seed)


def create_block_missing(p_fault=0.0015, p_noise=0.05, seed=42):
    """
    블록 결측치가 있는 PemsBay 데이터셋을 생성합니다.

    Args:
        p_fault: 연속적인 결측치를 생성하는 비율 (기본값: 0.0015)
        p_noise: 독립적인 결측치를 생성하는 비율 (기본값: 0.05)
        seed: 랜덤 시드 (기본값: 42)

    Returns:
        dataset: 블록 결측치가 추가된 PemsBay 데이터셋
    """
    return load_pemsbay_dataset(p_fault=p_fault, p_noise=p_noise, seed=seed)


if __name__ == "__main__":
    # 결측치가 있는 데이터셋 로드
    print("\n=== 블록 결측치 생성 ===")
    block_dataset = create_block_missing()

    print("\n=== 포인트 결측치 생성 ===")
    point_dataset = create_point_missing()

    # 데이터셋 저장
    print("\n=== 블록 결측치 데이터셋 저장 ===")
    save_pemsbay_data(block_dataset, output_dir="./bay/data/block")

    print("\n=== 포인트 결측치 데이터셋 저장 ===")
    save_pemsbay_data(point_dataset, output_dir="./bay/data/point")
