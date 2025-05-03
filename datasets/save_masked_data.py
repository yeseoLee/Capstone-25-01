#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
결측치가 있는 원본 데이터와 마스킹된 데이터를 함께 저장하는 스크립트

사용법:
    python save_masked_data.py --dataset_name bay_block --output_dir ./datasets/masked

이 스크립트는 원본 데이터셋을 로드하고, 결측치를 생성한 다음,
원본 데이터셋 형태(.h5 또는 .csv)로 다시 저장합니다.
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import sys
import h5py
import shutil

# 라이브러리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.data.imputation_dataset import GraphImputationDataset, ImputationDataset
from scripts.run_imputation import get_dataset
from lib import datasets_path


def parse_args():
    """
    명령행 인자를 파싱합니다.
    """
    parser = argparse.ArgumentParser(description="결측치가 있는 원본 데이터와 마스킹된 데이터를 함께 저장합니다.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bay_block",
        help="데이터셋 이름 (bay_block, la_block, bay_point, la_point)",
    )
    parser.add_argument("--output_dir", type=str, default="./datasets/masked", help="출력 디렉토리 경로")
    parser.add_argument("--window", type=int, default=24, help="윈도우 크기")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 스트라이드")
    parser.add_argument("--p_fault", type=float, default=0.0015, help="결함 확률 (연속적인 결측치를 생성하는 비율)")
    parser.add_argument("--p_noise", type=float, default=0.05, help="잡음 확률 (독립적인 결측치를 생성하는 비율)")
    parser.add_argument("--save_csv", type=bool, default=False, help="CSV 파일도 함께 저장할지 여부")

    return parser.parse_args()


def save_masked_data_csv(dataset, output_dir, file_prefix=None):
    """
    데이터셋의 마스킹된 데이터를 CSV 형식으로 저장합니다.

    Args:
        dataset: 데이터셋 객체 (ImputationDataset 또는 그 하위 클래스)
        output_dir: 출력 디렉토리 경로
        file_prefix: 출력 파일 접두사

    Returns:
        saved_files: 저장된 파일 경로 목록
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if file_prefix is None:
        file_prefix = "masked_data"

    # ImputationDataset 클래스의 save_masked_data 메서드를 사용하여 데이터 저장
    if hasattr(dataset, "save_masked_data"):
        saved_files = dataset.save_masked_data(output_dir, file_prefix)
        print(f"CSV 데이터가 {output_dir} 디렉토리에 저장되었습니다.")
        return saved_files
    else:
        raise TypeError("dataset은 save_masked_data 메서드를 가진 ImputationDataset 클래스여야 합니다.")


def create_masked_dataset(original_df, mask, dataset_name):
    """
    원본 데이터프레임과 마스크를 사용하여 결측치가 있는 데이터셋을 생성합니다.

    Args:
        original_df: 원본 데이터프레임
        mask: 마스크 (1: 유효한 데이터, 0: 결측치)
        dataset_name: 데이터셋 이름

    Returns:
        masked_df: 결측치가 포함된 데이터프레임
    """
    # 마스크가 0인 위치에 NaN 입력
    masked_df = original_df.copy()
    masked_values = masked_df.values
    masked_values[~mask.astype(bool)] = np.nan
    masked_df = pd.DataFrame(masked_values, index=original_df.index, columns=original_df.columns)

    return masked_df


def save_h5_dataset(df, output_file):
    """
    데이터프레임을 HDF5 형식으로 저장합니다.

    Args:
        df: 저장할 데이터프레임
        output_file: 출력 파일 경로
    """
    df.to_hdf(output_file, key="df")
    print(f"HDF5 데이터가 {output_file}에 저장되었습니다.")


def copy_auxiliary_files(dataset_name, original_dir, output_dir):
    """
    필요한 보조 파일(예: 거리 행렬, 센서 ID 등)을 복사합니다.

    Args:
        dataset_name: 데이터셋 이름
        original_dir: 원본 데이터 디렉토리
        output_dir: 출력 디렉토리
    """
    if "bay" in dataset_name:
        # PemsBay 데이터셋의 보조 파일 복사
        files_to_copy = ["pems_bay_dist.npy", "distances_bay.csv"]
        for file in files_to_copy:
            if os.path.exists(os.path.join(original_dir, file)):
                shutil.copy2(os.path.join(original_dir, file), os.path.join(output_dir, file))
                print(f"보조 파일 {file}이 복사되었습니다.")

    elif "la" in dataset_name:
        # MetrLA 데이터셋의 보조 파일 복사
        files_to_copy = ["metr_la_dist.npy", "distances_la.csv", "sensor_ids_la.txt"]
        for file in files_to_copy:
            if os.path.exists(os.path.join(original_dir, file)):
                shutil.copy2(os.path.join(original_dir, file), os.path.join(output_dir, file))
                print(f"보조 파일 {file}이 복사되었습니다.")


def main():
    """
    메인 함수
    """
    args = parse_args()

    try:
        # 데이터셋 로드
        print(f"데이터셋 '{args.dataset_name}' 로드 중...")
        dataset = get_dataset(args.dataset_name)

        # 데이터셋 정보 확인
        original_df = dataset.dataframe()
        mask = dataset.mask
        eval_mask = dataset.eval_mask

        # 출력 디렉토리 생성
        output_base_dir = Path(args.output_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # 데이터셋별 출력 디렉토리 생성 - dataset_name 전체를 사용
        dataset_output_dir = output_base_dir / args.dataset_name
        dataset_output_dir.mkdir(exist_ok=True)

        # 결측치가 있는 데이터셋 생성
        masked_df = create_masked_dataset(original_df, dataset.training_mask, args.dataset_name)

        # 데이터셋 형식에 맞게 저장
        if "bay" in args.dataset_name:
            # PemsBay 데이터셋은 h5 형식으로 저장
            save_h5_dataset(masked_df, os.path.join(dataset_output_dir, f"pems_bay_masked.h5"))
            # 보조 파일 복사
            copy_auxiliary_files(args.dataset_name, datasets_path["bay"], dataset_output_dir)

        elif "la" in args.dataset_name:
            # MetrLA 데이터셋은 h5 형식으로 저장
            save_h5_dataset(masked_df, os.path.join(dataset_output_dir, f"metr_la_masked.h5"))
            # 보조 파일 복사
            copy_auxiliary_files(args.dataset_name, datasets_path["la"], dataset_output_dir)

        # CSV 형식으로도 저장 (선택 사항)
        if args.save_csv:
            csv_output_dir = output_base_dir / f"{args.dataset_name}_csv"
            csv_output_dir.mkdir(exist_ok=True)

            # 마스킹된 데이터셋 생성
            has_graph_support = True  # GRIN 모델은 그래프 지원이 필요
            dataset_cls = GraphImputationDataset if has_graph_support else ImputationDataset

            torch_dataset = dataset_cls(
                *dataset.numpy(return_idx=True),
                mask=dataset.training_mask,
                eval_mask=dataset.eval_mask,
                window=args.window,
                stride=args.stride,
            )

            # CSV 형식으로 저장
            print(f"마스킹된 데이터를 CSV 형식으로 '{csv_output_dir}' 디렉토리에 저장 중...")
            saved_files = save_masked_data_csv(torch_dataset, csv_output_dir, args.dataset_name)

            print(f"\n저장된 CSV 파일 개수: {len(saved_files)}개")

        print(f"\n모든 파일이 {output_base_dir} 디렉토리에 성공적으로 저장되었습니다.")
        print(f"사용된 마스크: 결함 확률={args.p_fault}, 잡음 확률={args.p_noise}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
