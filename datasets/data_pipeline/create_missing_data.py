"""
STGAN 데이터셋에 결측치(마스킹)를 생성하고 저장하는 스크립트

사용법:
    # 기본 사용법 (전체 구간에 결측치 생성)
    python ./datasets/data_pipeline/create_missing_data.py --dataset_name bay --output_dir ./datasets/bay/masked --p_fault 0.0015 --p_noise 0.05

    # 특정 구간에만 결측치 생성
    python ./datasets/data_pipeline/create_missing_data.py --dataset_name bay --output_dir ./datasets/bay/masked --p_fault 0.0015 --p_noise 0.05 --start_interval 0.2 --end_interval 0.8

    # point 방식으로 결측치 생성 (독립적인 결측치만 생성)
    python ./datasets/data_pipeline/create_missing_data.py --dataset_name bay --output_dir ./datasets/bay/masked --mask_type point

파라미터 설명:
    --dataset_name: 데이터셋 이름 (현재는 'bay'만 지원)
    --output_dir: 출력 디렉토리 경로
    --p_fault: 결함 확률 (연속적인 결측치를 생성하는 비율, 기본값: 0.0015)
    --p_noise: 잡음 확률 (독립적인 결측치를 생성하는 비율, 기본값: 0.05)
    --mask_type: 마스크 유형 ('block': 연속적 결측치, 'point': 독립적 결측치)
    --start_interval: 결측치 생성 시작 구간 (0.0~1.0, 기본값: 0.0)
    --end_interval: 결측치 생성 종료 구간 (0.0~1.0, 기본값: 1.0)

예시:
    # 전체 구간의 20%에서 80% 사이에만 결측치 생성
    python ./datasets/data_pipeline/create_missing_data.py --dataset_name bay --output_dir ./datasets/bay/masked --start_interval 0.2 --end_interval 0.8

    # 전체 구간의 50% 이후부터 결측치 생성
    python ./datasets/data_pipeline/create_missing_data.py --dataset_name bay --output_dir ./datasets/bay/masked --start_interval 0.5
"""  # noqa: E501

import argparse
import os
from pathlib import Path

import numpy as np


# STGAN 데이터 경로 정의
STGAN_DATA_PATH = "./datasets/bay/data"


def create_missing_mask(  # noqa: C901
    data_shape, p_fault=0.0015, p_noise=0.05, mask_type="block", start_interval=0.0, end_interval=1.0
):
    """
    결측치 마스크 생성

    Args:
        data_shape: 데이터 형태 (시간, 노드, 특징1, 특징2)
        p_fault: 결함 확률 (연속적인 결측치를 생성하는 비율)
        p_noise: 잡음 확률 (독립적인 결측치를 생성하는 비율)
        mask_type: 마스크 유형 ("block" 또는 "point")
        start_interval: 결측치 생성 시작 구간 (0.0~1.0)
        end_interval: 결측치 생성 종료 구간 (0.0~1.0)

    Returns:
        mask: 마스크 (1: 유효한 데이터, 0: 결측치)
    """
    # 마스크 초기화 (1: 유효한 데이터)
    mask = np.ones(data_shape)

    # 시작 및 종료 구간 계산
    time_len = data_shape[0]
    start_idx = int(time_len * start_interval)
    end_idx = int(time_len * end_interval)

    if mask_type == "point":
        pass
        # point 방식: 독립적인 결측치만 생성
        # p_fault = 0.0
        # p_noise = 0.1  # 기본값 증가

    # 결함 (연속적인 결측치) 생성
    if p_fault > 0:
        for i in range(data_shape[1]):  # 노드별 처리
            for t in range(start_idx, end_idx):  # 지정된 구간에서만 시간별 처리
                if np.random.random() < p_fault:
                    # 결함 길이 (1~10 사이 임의 값)
                    fault_length = np.random.randint(1, 11)
                    if t + fault_length <= end_idx:
                        # 모든 특성에 동일한 마스크 적용
                        mask[t : t + fault_length, i, :, :] = 0

    # 잡음 (독립적인 결측치) 생성
    if p_noise > 0:
        # 노드 및 시간에 대해 랜덤 마스크 생성
        random_mask = np.random.random(data_shape[:2]) < p_noise
        for t in range(start_idx, end_idx):
            for i in range(data_shape[1]):
                if random_mask[t, i]:
                    # 모든 특성에 동일한 마스크 적용
                    mask[t, i, :, :] = 0

    return mask


def save_masked_data(data, mask, output_dir, mask_type="block", p_fault=0.0015, p_noise=0.05, start_interval=0.0, end_interval=1.0):
    """
    마스킹된 데이터를 저장

    Args:
        data: 원본 데이터
        mask: 마스크 (1: 유효한 데이터, 0: 결측치)
        output_dir: 출력 디렉토리
        mask_type: 마스크 유형
        p_fault: 결함 확률
        p_noise: 잡음 확률
        start_interval: 결측치 생성 시작 구간
        end_interval: 결측치 생성 종료 구간
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 폴더명에 파라미터 포함
    folder_name = f"{mask_type}_fault{p_fault:.4f}_noise{p_noise:.4f}_interval{start_interval:.2f}-{end_interval:.2f}"
    data_dir = output_path / folder_name
    data_dir.mkdir(parents=True, exist_ok=True)

    # 마스킹된 데이터 생성 (결측치는 NaN으로 표시)
    masked_data = data.copy()
    masked_data[mask == 0] = np.nan

    # 데이터 저장
    np.save(os.path.join(data_dir, "data.npy"), masked_data)
    np.save(os.path.join(data_dir, "mask.npy"), mask)  # 마스크도 저장

    # 필요한 보조 파일 복사
    for file in ["time_features.txt", "node_subgraph.npy", "node_adjacent.txt", "node_dist.txt"]:
        src_file = os.path.join(STGAN_DATA_PATH, file)
        if os.path.exists(src_file):
            import shutil

            shutil.copy2(src_file, os.path.join(data_dir, file))

    print(f"마스킹된 데이터가 {data_dir}에 저장되었습니다.")
    return data_dir


def create_masked_dataset(dataset_name, output_dir, p_fault=0.0015, p_noise=0.05, mask_type="block", start_interval=0.0, end_interval=1.0):
    """
    마스킹된 데이터셋 생성

    Args:
        dataset_name: 데이터셋 이름
        output_dir: 출력 디렉토리
        p_fault: 결함 확률
        p_noise: 잡음 확률
        mask_type: 마스크 유형 ("block" 또는 "point")
        start_interval: 결측치 생성 시작 구간
        end_interval: 결측치 생성 종료 구간
    """
    if dataset_name == "bay":
        data_path = os.path.join(STGAN_DATA_PATH, "data.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

        # 데이터 로드
        print(f"데이터셋 '{dataset_name}' 로드 중...")
        data = np.load(data_path)
        print(f"데이터 형태: {data.shape}")

        # 마스크 생성
        print(
            f"마스크 생성 중 (타입: {mask_type}, 결함: {p_fault}, 잡음: {p_noise}, 구간: {start_interval:.2f}-{end_interval:.2f})..."  # noqa: E501
        )
        mask = create_missing_mask(data.shape, p_fault, p_noise, mask_type, start_interval, end_interval)

        # 마스킹된 데이터 저장
        data_dir = save_masked_data(data, mask, output_dir, mask_type, p_fault, p_noise, start_interval, end_interval)

        # 통계 정보 출력
        missing_ratio = 1.0 - np.mean(mask)
        print("생성된 마스크 정보:")
        print(f"- 결측치 비율: {missing_ratio:.4f} ({missing_ratio * 100:.2f}%)")
        print(f"- 마스크 형태: {mask.shape}")

        return data_dir
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {dataset_name}")


def main(args):
    """
    메인 함수
    """
    try:
        # 출력 디렉토리 생성
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 마스킹된 데이터셋 생성
        data_dir = create_masked_dataset(
            args.dataset_name,
            output_dir,
            args.p_fault,
            args.p_noise,
            args.mask_type,
            args.start_interval,
            args.end_interval,
        )

        print(f"\n모든 파일이 {data_dir} 디렉토리에 성공적으로 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGAN 데이터셋에 결측치를 생성하는 스크립트")
    parser.add_argument("--dataset_name", type=str, default="bay", help="데이터셋 이름 (현재는 bay만 지원)")
    parser.add_argument("--output_dir", type=str, default="./datasets/bay/masked", help="출력 디렉토리 경로")
    parser.add_argument("--p_fault", type=float, default=0.0015, help="결함 확률 (연속적인 결측치를 생성하는 비율)")
    parser.add_argument("--p_noise", type=float, default=0.05, help="잡음 확률 (독립적인 결측치를 생성하는 비율)")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="block",
        choices=["block", "point"],
        help="마스크 유형 (block: 연속적 결측치, point: 독립적 결측치)",
    )
    parser.add_argument(
        "--start_interval",
        type=float,
        default=0.0,
        help="결측치 생성 시작 구간 (0.0~1.0)",
    )
    parser.add_argument(
        "--end_interval",
        type=float,
        default=1.0,
        help="결측치 생성 종료 구간 (0.0~1.0)",
    )

    args = parser.parse_args()
    main(args)
