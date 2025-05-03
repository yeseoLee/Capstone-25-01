import argparse
import os

import numpy as np
import pandas as pd
from tsl.datasets import MetrLA, PemsBay


def download_datasets(datasets):
    """
    TSL 라이브러리를 사용하여 교통 데이터셋을 다운로드합니다.

    Args:
        datasets: 다운로드할 데이터셋 목록 ('metr_la', 'pems_bay' 또는 둘 다)
    """
    for dataset_name in datasets:
        # 데이터셋 폴더 생성
        os.makedirs(f"./datasets/{dataset_name}", exist_ok=True)

        if dataset_name == "metr_la":
            # METR-LA 데이터셋 다운로드
            dataset = MetrLA(root=f"./datasets/{dataset_name}")
            print("METR-LA 데이터셋 다운로드 완료!")
            print(f"데이터셋 크기: {dataset.target.shape}")

        elif dataset_name == "pems_bay":
            # PEMS-Bay 데이터셋 다운로드
            dataset = PemsBay(root=f"./datasets/{dataset_name}")
            print("PEMS-Bay 데이터셋 다운로드 완료!")
            print(f"데이터셋 크기: {dataset.target.shape}")

    print("데이터셋 다운로드 완료!")


def extract_files(datasets):
    """
    locations.csv와 {dataset_name}_dist.npy 파일로부터
    distances.csv와 sensor_ids.txt 파일을 생성합니다.

    Args:
        datasets: 파일을 추출할 데이터셋 목록 ('metr_la', 'pems_bay' 또는 둘 다)
    """
    for dataset_name in datasets:
        base_dir = f"datasets/{dataset_name}"
        dist_file = f"{base_dir}/{dataset_name}_dist.npy"
        locations_file = f"{base_dir}/locations.csv"

        # 파일이 존재하는지 확인
        if not os.path.exists(dist_file):
            print(f"Error: {dist_file} 파일이 존재하지 않습니다.")
            continue

        if not os.path.exists(locations_file):
            print(f"Error: {locations_file} 파일이 존재하지 않습니다.")
            continue

        # 거리 행렬 로드
        dist_matrix = np.load(dist_file)

        # 위치 데이터 로드 - 데이터셋별 형식 차이 처리
        locations = pd.read_csv(locations_file)

        # 데이터셋별 locations.csv 형식 차이 처리
        if dataset_name == "metr_la":
            sensor_ids = locations["sensor_id"].astype(str).values
        else:  # pems_bay
            # PEMS Bay 데이터셋은 열 이름이 없고 첫 번째 열이 센서 ID
            if "sensor_id" not in locations.columns:
                # 열 이름이 없는 경우 첫 번째 열을 센서 ID로 사용
                sensor_ids = locations.iloc[:, 0].astype(str).values

        # sensor_ids.txt 생성
        dataset_suffix = dataset_name.split("_")[-1]  # la 또는 bay
        with open(f"{base_dir}/sensor_ids_{dataset_suffix}.txt", "w") as f:
            f.write("\n".join(sensor_ids))

        # distances.csv 생성
        n = dist_matrix.shape[0]
        with open(f"{base_dir}/distances_{dataset_suffix}.csv", "w") as f:
            # CSV 헤더 작성 (from, to, cost)
            f.write("from,to,cost\n")
            for i in range(n):
                for j in range(n):
                    # inf가 아닌 값만 저장 (즉, 연결된 노드만)
                    if not np.isinf(dist_matrix[i, j]) and i != j:
                        f.write(f"{i},{j},{dist_matrix[i, j]}\n")

        print(f"{dataset_name} 데이터셋에 대한 파일 생성 완료:")
        print(f"- sensor_ids_{dataset_suffix}.txt")
        print(f"- distances_{dataset_suffix}.csv")

    print("모든 파일 추출 작업이 완료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="교통 데이터셋(METR-LA, PEMS-BAY) 준비 도구")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["metr_la", "pems_bay"],
        help="처리할 데이터셋 ('metr_la', 'pems_bay' 또는 둘 다)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="데이터셋만 다운로드하고 파일 추출은 하지 않음",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="파일 추출만 수행하고 데이터셋 다운로드는 하지 않음",
    )

    args = parser.parse_args()

    # 유효한 데이터셋 이름인지 확인
    valid_datasets = ["metr_la", "pems_bay"]
    datasets = [d for d in args.datasets if d in valid_datasets]

    if not datasets:
        print("오류: 유효한 데이터셋을 지정해야 합니다 ('metr_la', 'pems_bay')")
        return

    # 다운로드 및 추출 작업 수행
    if not args.extract_only:
        print("=== 데이터셋 다운로드 시작 ===")
        download_datasets(datasets)
        print()

    if not args.download_only:
        print("=== 파일 추출 시작 ===")
        extract_files(datasets)


if __name__ == "__main__":
    main()
