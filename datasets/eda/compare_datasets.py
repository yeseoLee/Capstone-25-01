import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tsl.datasets import PemsBay


# 폰트 설정 - 한글 지원
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_stgan_data(data_dir):
    """STGAN 데이터 로드 함수"""
    print("STGAN 데이터셋 로드 중...")

    # 데이터 파일 경로
    data_path = os.path.join(data_dir, "data.npy")
    time_features_path = os.path.join(data_dir, "time_features.txt")
    node_adjacent_path = os.path.join(data_dir, "node_adjacent.txt")

    # 데이터 로드
    data = np.load(data_path)
    time_features = np.loadtxt(time_features_path)

    # 노드 인접 정보 로드 (파일이 클 수 있으므로 첫 10개 라인만 읽기)
    with open(node_adjacent_path, "r") as f:
        node_adjacent_sample = [next(f) for _ in range(10)]

    return {"data": data, "time_features": time_features, "node_adjacent_sample": node_adjacent_sample}


def load_pemsbay_data():
    """TSL PemsBay 데이터셋 로드 함수"""
    print("TSL PemsBay 데이터셋 로드 중...")

    # PemsBay 데이터셋 로드
    dataset = PemsBay()

    # 데이터 추출
    data = dataset.numpy()

    # 연결성 행렬 추출
    connectivity = dataset.get_connectivity(threshold=0.1, include_self=False)
    edge_index, edge_weight = connectivity

    # 인접 행렬 생성
    adj_matrix = np.zeros((dataset.n_nodes, dataset.n_nodes))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i, j] = 1

    return {"dataset": dataset, "data": data, "adj_matrix": adj_matrix}


def compare_and_visualize():
    """두 데이터셋을 비교하고 시각화"""
    # 데이터 로드
    stgan_data_dir = os.path.join("datasets", "bay", "data")

    try:
        stgan_data = load_stgan_data(stgan_data_dir)
        pemsbay_data = load_pemsbay_data()

        # 기본 정보 출력
        # print("\n=== STGAN 데이터셋 정보 ===")
        # print(f"데이터 형상: {stgan_data['data'].shape}")
        # print(f"시간 특성 형상: {stgan_data['time_features'].shape}")
        # print(f"노드 인접 정보 샘플: {stgan_data['node_adjacent_sample'][0][:100]}...")

        # print("\n=== TSL PemsBay 데이터셋 정보 ===")
        # print(f"데이터 형상: {pemsbay_data['data'].shape}")
        # print(f"연결성 행렬 형상: {pemsbay_data['adj_matrix'].shape}")

        # 데이터 구조 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. STGAN 데이터 히스토그램
        sample_data = stgan_data["data"].reshape(-1)[:10000]
        sns.histplot(sample_data[~np.isnan(sample_data)], bins=50, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("STGAN 데이터 분포 (샘플)")
        axes[0, 0].set_xlabel("값")
        axes[0, 0].set_ylabel("빈도")

        # 2. STGAN 시간별 평균 값
        stgan_time_avg = np.nanmean(stgan_data["data"], axis=(1, 2, 3))
        axes[0, 1].plot(stgan_time_avg[:288])  # 첫 24시간(12 * 24 = 288) 데이터만 표시
        axes[0, 1].set_title("STGAN 시간별 평균값 (첫 24시간)")
        axes[0, 1].set_xlabel("시간 단계")
        axes[0, 1].set_ylabel("평균값")

        # 3. STGAN 시간 특성 히트맵
        sns.heatmap(stgan_data["time_features"][:24], cmap="viridis", ax=axes[0, 2])
        axes[0, 2].set_title("STGAN 시간 특성 (첫 24 시간)")
        axes[0, 2].set_xlabel("특성 차원")
        axes[0, 2].set_ylabel("시간 단계")

        # 4. TSL PemsBay 데이터 히스토그램
        sample_data = pemsbay_data["data"].reshape(-1)[:10000]
        sns.histplot(sample_data[~np.isnan(sample_data)], bins=50, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("TSL PemsBay 데이터 분포 (샘플)")
        axes[1, 0].set_xlabel("값")
        axes[1, 0].set_ylabel("빈도")

        # 5. TSL PemsBay 시간별 평균 값
        pemsbay_time_avg = np.nanmean(pemsbay_data["data"], axis=(1, 2))
        axes[1, 1].plot(pemsbay_time_avg[:288])  # 첫 24시간 데이터만 표시
        axes[1, 1].set_title("TSL PemsBay 시간별 평균값 (첫 24시간)")
        axes[1, 1].set_xlabel("시간 단계")
        axes[1, 1].set_ylabel("평균값")

        # 6. TSL PemsBay 연결성 행렬 시각화
        sns.heatmap(pemsbay_data["adj_matrix"][:50, :50], cmap="Blues", ax=axes[1, 2])
        axes[1, 2].set_title("TSL PemsBay 연결성 행렬 (50x50 부분)")
        axes[1, 2].set_xlabel("노드 ID")
        axes[1, 2].set_ylabel("노드 ID")

        plt.tight_layout()
        plt.savefig("datasets/dataset_comparison.png", dpi=300)

        # 데이터 차원 비교표 생성
        compare_data = {
            "데이터셋": ["STGAN", "TSL PemsBay"],
            "데이터 차원": [f"{stgan_data['data'].shape}", f"{pemsbay_data['data'].shape}"],
        }

        compare_df = pd.DataFrame(compare_data)
        print("\n=== 데이터셋 차원 비교 ===")
        print(compare_df.to_string(index=False))

        # STGAN 4D -> 3D 변환 예시
        if len(stgan_data["data"].shape) == 4:
            n_time, n_nodes, n_features, n_channels = stgan_data["data"].shape
            stgan_3d = stgan_data["data"].reshape(n_time, n_nodes, n_features * n_channels)
            print(f"\nSTGAN 4D -> 3D 변환 결과: {stgan_3d.shape}")

            # STGAN 3D와 TSL PemsBay 데이터 채널 수 비교
            print(f"STGAN 변환 후 채널 수: {stgan_3d.shape[2]}")
            print(f"TSL PemsBay 채널 수: {pemsbay_data['data'].shape[2]}")

            # STGAN 3D와 TSL PemsBay 데이터 비교 시각화
            plt.figure(figsize=(12, 6))

            # 첫 번째 노드 데이터의 첫 시간 단계 데이터를 비교
            plt.subplot(1, 2, 1)
            if np.any(~np.isnan(stgan_3d[0, 0])):
                plt.plot(stgan_3d[0, 0], label="STGAN 변환 후")
                plt.title("STGAN 3D 변환 후 첫 번째 노드 데이터")
                plt.xlabel("채널")
                plt.ylabel("값")
                plt.legend()
            else:
                plt.text(0.5, 0.5, "NaN 값", horizontalalignment="center")
                plt.title("STGAN 첫 번째 노드 데이터 (NaN)")

            plt.subplot(1, 2, 2)
            if np.any(~np.isnan(pemsbay_data["data"][0, 0])):
                plt.plot(pemsbay_data["data"][0, 0], label="TSL PemsBay")
                plt.title("TSL PemsBay 첫 번째 노드 데이터")
                plt.xlabel("채널")
                plt.ylabel("값")
                plt.legend()
            else:
                plt.text(0.5, 0.5, "NaN 값", horizontalalignment="center")
                plt.title("TSL PemsBay 첫 번째 노드 데이터 (NaN)")

            plt.tight_layout()
            plt.savefig("datasets/node_data_comparison.png", dpi=300)

            # 간단한 시각화: 두 데이터셋의 구조 차이 요약
            fig, ax = plt.subplots(figsize=(12, 8))

            # 두 데이터셋 구조 비교를 위한 테이블 데이터
            table_data = [
                ["차원수", "4차원 (시간, 노드, 특성, 채널)", "3차원 (시간, 노드, 채널)"],
                ["데이터 형상", f"{stgan_data['data'].shape}", f"{pemsbay_data['data'].shape}"],
                ["노드 수", f"{stgan_data['data'].shape[1]}", f"{pemsbay_data['data'].shape[1]}"],
                ["변환 형태", f"4D -> 3D: {stgan_3d.shape}", "N/A"],
                ["시간 특성", "time_features.txt (31차원)", "시간/요일 인코딩 내장"],
                ["연결 정보", "node_adjacent.txt", "연결성 행렬"],
            ]

            ax.axis("tight")
            ax.axis("off")
            table = ax.table(
                cellText=table_data,
                colLabels=["항목", "STGAN 데이터셋", "TSL PemsBay 데이터셋"],
                loc="center",
                cellLoc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)

            plt.title("STGAN과 TSL PemsBay 데이터셋 구조 비교", fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig("datasets/dataset_structure_comparison.png", dpi=300)

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    compare_and_visualize()
