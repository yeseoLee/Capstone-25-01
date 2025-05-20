"""
STGAN 모델을 사용하여 이상치를 탐지하고 변환하는 스크립트
"""

import argparse
import os

from load_data import DataLoader
import numpy as np
from tester import Tester


def detect_and_convert_outliers(args):
    """이상치를 탐지하고 변환합니다."""
    # 데이터 로더 초기화
    loader = DataLoader(args)

    # 테스터 초기화
    tester = Tester(args)

    # 이상치 탐지 및 변환
    print("이상치 탐지 및 변환 중...")
    converted_data = tester.detect_and_convert_outliers()

    # 결과 저장
    output_dir = os.path.join(args.root_path, f"datasets/{args.dataset}/anomaly_data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "data.npy")
    np.save(output_path, converted_data)
    print(f"변환된 데이터가 저장되었습니다: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bay", help="데이터셋 이름")
    parser.add_argument("--root_path", type=str, default="./", help="루트 경로")
    parser.add_argument("--outlier_threshold", type=float, default=0.1, help="이상치 탐지 임계값")
    parser.add_argument("--use-node-subset", action="store_true", help="노드 서브셋 사용")
    parser.add_argument("--node-list", type=str, help="사용할 노드 인덱스 목록 (쉼표로 구분)")

    args = parser.parse_args()
    detect_and_convert_outliers(args)
