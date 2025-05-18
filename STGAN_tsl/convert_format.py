"""
PemsBay 데이터셋을 STGAN 형식으로 변환하는 모듈

이 모듈은 TSL의 PemsBay 데이터셋을 STGAN에서 사용할 수 있는 형식으로
변환하는 기능을 제공합니다.

기본 사용법:
    python convert_format.py
"""

import os

import numpy as np
from tsl.datasets import PemsBay


def pemsbay_to_stgan_format(output_dir="./bay/data/stgan_format"):
    """
    PemsBay 데이터셋을 STGAN 형식으로 변환하여 저장합니다.

    STGAN 형식:
    - data.npy: (시간, 노드, 특성, 채널) 4D 배열
    - time_features.txt: 시간 특성 (요일 + 시간)
    - node_adjacent.txt: 노드 인접 정보
    - node_dist.txt: 노드 간 거리 정보

    Args:
        output_dir: 출력 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    # PemsBay 데이터셋 로드
    print("PemsBay 데이터셋 로드 중...")
    dataset = PemsBay()

    print(f"데이터셋 크기: {dataset.shape}")
    print(f"노드 수: {dataset.n_nodes}")

    # 데이터 추출
    data = dataset.numpy()  # (시간, 노드, 채널)

    # STGAN 형식으로 변환 (시간, 노드, 특성, 채널)
    # PemsBay는 채널이 1개이므로, 특성을 1로 설정
    stgan_data = data.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
    print(f"STGAN 형식 데이터 크기: {stgan_data.shape}")

    # 데이터 저장
    np.save(os.path.join(output_dir, "data.npy"), stgan_data)

    # 시간 특성 저장 (요일 + 시간)
    time_feat = dataset.datetime_encoded(["day", "week"]).values
    np.savetxt(os.path.join(output_dir, "time_features.txt"), time_feat)

    # 인접 행렬 계산
    adj = dataset.get_connectivity(threshold=0.1, include_self=False)

    # 인접 행렬을 STGAN 형식으로 변환
    # 노드 인접 정보 저장
    node_adjacent = []
    for i in range(adj.shape[0]):
        neighbors = np.where(adj[i] > 0)[0]
        neighbors_str = ",".join(map(str, neighbors))
        node_adjacent.append(f"{i}:{neighbors_str}")

    with open(os.path.join(output_dir, "node_adjacent.txt"), "w") as f:
        f.write("\n".join(node_adjacent))

    # 노드 간 거리 저장 (실제 거리 정보가 없으므로 인접 행렬 기반 근사값 사용)
    node_dist = []
    for i in range(adj.shape[0]):
        neighbors = np.where(adj[i] > 0)[0]
        distances = np.ones_like(neighbors)  # 모든 인접 노드 간 거리를 1로 설정
        dist_str = ",".join([f"{n}:{d}" for n, d in zip(neighbors, distances)])
        node_dist.append(f"{i}:{dist_str}")

    with open(os.path.join(output_dir, "node_dist.txt"), "w") as f:
        f.write("\n".join(node_dist))

    # 서브그래프 정보 생성 (모든 노드를 하나의 서브그래프로 처리)
    subgraph = np.zeros((dataset.n_nodes, 1), dtype=int)
    np.save(os.path.join(output_dir, "node_subgraph.npy"), subgraph)

    print(f"PemsBay 데이터셋이 STGAN 형식으로 {output_dir}에 저장되었습니다.")
    print("파일 목록:")
    print(f"- data.npy: 데이터 배열 (크기: {stgan_data.shape})")
    print(f"- time_features.txt: 시간 특성 (크기: {time_feat.shape})")
    print("- node_adjacent.txt: 노드 인접 정보")
    print("- node_dist.txt: 노드 간 거리 정보")
    print(f"- node_subgraph.npy: 노드 서브그래프 정보 (크기: {subgraph.shape})")


def pemsbay_with_missing_to_stgan_format(
    p_fault=0.0015, p_noise=0.05, seed=42, output_dir="./bay/data/stgan_format_missing"
):
    """
    결측치가 있는 PemsBay 데이터셋을 STGAN 형식으로 변환하여 저장합니다.

    Args:
        p_fault: 연속적인 결측치를 생성하는 비율 (기본값: 0.0015)
        p_noise: 독립적인 결측치를 생성하는 비율 (기본값: 0.05)
        seed: 랜덤 시드 (기본값: 42)
        output_dir: 출력 디렉토리 경로
    """
    from tsl.ops.imputation import add_missing_values

    os.makedirs(output_dir, exist_ok=True)

    # PemsBay 데이터셋 로드
    print("PemsBay 데이터셋 로드 중...")
    dataset = PemsBay()

    # 결측치 추가
    print(f"결측치 추가 중... (p_fault={p_fault}, p_noise={p_noise}, seed={seed})")
    dataset_with_missing = add_missing_values(
        dataset, p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=seed
    )

    # 결측치 통계
    missing_ratio = 1.0 - np.mean(dataset_with_missing.training_mask)
    print(f"추가된 결측치 비율: {missing_ratio:.4f} ({missing_ratio*100:.2f}%)")

    # 데이터 및 마스크 추출
    data, mask = dataset_with_missing.numpy(return_mask=True)

    # 결측치가 있는 데이터 생성 (마스크가 False인 위치에 NaN 설정)
    data_with_nan = data.copy()
    data_with_nan[~mask] = np.nan

    # STGAN 형식으로 변환
    stgan_data = data_with_nan.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
    stgan_mask = mask.reshape(mask.shape[0], mask.shape[1], 1, mask.shape[2])

    # 데이터 저장
    np.save(os.path.join(output_dir, "data.npy"), stgan_data)
    np.save(os.path.join(output_dir, "mask.npy"), stgan_mask)

    # 나머지 파일들도 생성 (시간 특성, 인접 정보 등)
    time_feat = dataset.datetime_encoded(["day", "week"]).values
    np.savetxt(os.path.join(output_dir, "time_features.txt"), time_feat)

    # 인접 행렬
    adj = dataset.get_connectivity(threshold=0.1, include_self=False)

    # 인접 정보 저장
    node_adjacent = []
    for i in range(adj.shape[0]):
        neighbors = np.where(adj[i] > 0)[0]
        neighbors_str = ",".join(map(str, neighbors))
        node_adjacent.append(f"{i}:{neighbors_str}")

    with open(os.path.join(output_dir, "node_adjacent.txt"), "w") as f:
        f.write("\n".join(node_adjacent))

    # 거리 정보 저장
    node_dist = []
    for i in range(adj.shape[0]):
        neighbors = np.where(adj[i] > 0)[0]
        distances = np.ones_like(neighbors)
        dist_str = ",".join([f"{n}:{d}" for n, d in zip(neighbors, distances)])
        node_dist.append(f"{i}:{dist_str}")

    with open(os.path.join(output_dir, "node_dist.txt"), "w") as f:
        f.write("\n".join(node_dist))

    # 서브그래프 정보
    subgraph = np.zeros((dataset.n_nodes, 1), dtype=int)
    np.save(os.path.join(output_dir, "node_subgraph.npy"), subgraph)

    print(f"결측치가 있는 PemsBay 데이터셋이 STGAN 형식으로 {output_dir}에 저장되었습니다.")
    print("파일 목록:")
    print(f"- data.npy: 데이터 배열 (크기: {stgan_data.shape})")
    print(f"- mask.npy: 마스크 배열 (크기: {stgan_mask.shape})")
    print(f"- time_features.txt: 시간 특성 (크기: {time_feat.shape})")
    print("- node_adjacent.txt: 노드 인접 정보")
    print("- node_dist.txt: 노드 간 거리 정보")
    print(f"- node_subgraph.npy: 노드 서브그래프 정보 (크기: {subgraph.shape})")


if __name__ == "__main__":
    # 원본 PemsBay 데이터셋을 STGAN 형식으로 변환
    print("\n=== 원본 PemsBay 데이터셋을 STGAN 형식으로 변환 ===")
    pemsbay_to_stgan_format()

    # 블록 결측치가 있는 PemsBay 데이터셋을 STGAN 형식으로 변환
    print("\n=== 블록 결측치가 있는 PemsBay 데이터셋을 STGAN 형식으로 변환 ===")
    pemsbay_with_missing_to_stgan_format(p_fault=0.0015, p_noise=0.05, output_dir="./bay/data/stgan_format_block")

    # 포인트 결측치가 있는 PemsBay 데이터셋을 STGAN 형식으로 변환
    print("\n=== 포인트 결측치가 있는 PemsBay 데이터셋을 STGAN 형식으로 변환 ===")
    pemsbay_with_missing_to_stgan_format(p_fault=0.0, p_noise=0.25, output_dir="./bay/data/stgan_format_point")
