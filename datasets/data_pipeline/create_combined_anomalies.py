"""
STGAN 데이터셋에 결측치와 이상치를 동시에 생성하고 저장하는 스크립트

사용법:
    # 기본 사용법 (전체 구간에 결측치와 이상치 생성)
    python ./datasets/data_pipeline/create_combined_anomalies.py --dataset_name bay --output_dir ./datasets/bay/combined

    # 특정 구간에만 결측치와 이상치 생성
    python ./datasets/data_pipeline/create_combined_anomalies.py --dataset_name bay --output_dir ./datasets/bay/combined --start_interval 0.2 --end_interval 0.8

파라미터 설명:
    --dataset_name: 데이터셋 이름 (현재는 'bay'만 지원)
    --output_dir: 출력 디렉토리 경로
    --scenario: 이상치 생성 시나리오 ('point', 'block', 'contextual')
    --p_fault: 결함 확률 (연속적인 결측치를 생성하는 비율, 기본값: 0.1)
    --p_noise: 잡음 확률 (독립적인 결측치를 생성하는 비율, 기본값: 0.1)
    --mask_type: 마스크 유형 ('block': 연속적 결측치, 'point': 독립적 결측치)
    --min_deviation: 최소 편차 (정규화된 값, 기본값: 0.2)
    --max_deviation: 최대 편차 (정규화된 값, 기본값: 0.5)
    --p_outlier: 이상치 생성 확률 (기본값: 0.1)
    --start_interval: 이상치/결측치 생성 시작 구간 (0.0~1.0, 기본값: 0.0)
    --end_interval: 이상치/결측치 생성 종료 구간 (0.0~1.0, 기본값: 1.0)
    --seed: 랜덤 시드 (기본값: 42)

예시:
    # 전체 구간의 20%에서 80% 사이에만 결측치와 이상치 생성
    python ./datasets/data_pipeline/create_combined_anomalies.py --dataset_name bay --output_dir ./datasets/bay/combined --start_interval 0.2 --end_interval 0.8

    # 전체 구간의 50% 이후부터 결측치와 이상치 생성
    python ./datasets/data_pipeline/create_combined_anomalies.py --dataset_name bay --output_dir ./datasets/bay/combined --start_interval 0.5
"""  # noqa: E501

import argparse
import os
from pathlib import Path
import shutil

from create_missing_data import create_missing_mask
from create_outliers import get_outlier_generator
import numpy as np


# STGAN 데이터 경로 정의
STGAN_DATA_PATH = "./datasets/bay/data"


class CombinedAnomalyGenerator:
    """
    결측치와 이상치를 동시에 생성하는 클래스
    """

    def __init__(
        self,
        data,
        time_features,
        scenario="point",
        p_fault=0.1,
        p_noise=0.1,
        mask_type="block",
        min_deviation=0.2,
        max_deviation=0.5,
        p_outlier=0.1,
        start_interval=0.0,
        end_interval=1.0,
        seed=42,
    ):
        """
        초기화
        Args:
            data: 원본 데이터
            time_features: 시간 특성 데이터
            scenario: 이상치 생성 시나리오
            p_fault: 결함 확률
            p_noise: 잡음 확률
            mask_type: 마스크 유형
            min_deviation: 최소 편차
            max_deviation: 최대 편차
            p_outlier: 이상치 생성 확률
            start_interval: 생성 시작 구간
            end_interval: 생성 종료 구간
            seed: 랜덤 시드
        """
        self.data = data.copy()
        self.time_features = time_features
        self.scenario = scenario
        self.p_fault = p_fault
        self.p_noise = p_noise
        self.mask_type = mask_type
        self.min_deviation = min_deviation
        self.max_deviation = max_deviation
        self.p_outlier = p_outlier
        self.start_interval = start_interval
        self.end_interval = end_interval
        self.seed = seed

        # 랜덤 시드 설정
        np.random.seed(seed)

        # 시작 및 종료 구간 계산
        time_len = self.data.shape[0]
        self.start_idx = int(time_len * start_interval)
        self.end_idx = int(time_len * end_interval)

        # 마스크 초기화 (True: 정상, False: 이상치/결측치)
        self.outlier_mask = np.ones_like(data, dtype=bool)
        self.missing_mask = np.ones_like(data, dtype=bool)

    def generate(self):
        """
        결측치와 이상치 생성
        """
        print("결측치와 이상치 생성 중...")

        # 1. 먼저 결측치 마스크 생성
        missing_mask = create_missing_mask(
            self.data.shape,
            self.p_fault,
            self.p_noise,
            self.mask_type,
            self.start_interval,
            self.end_interval,
        )
        self.missing_mask = missing_mask.astype(bool)

        # 2. 결측치가 아닌 영역에만 이상치 생성
        # 결측치가 아닌 영역의 마스크 생성
        valid_for_outliers = self.missing_mask.copy()

        # 이상치 생성기 초기화
        outlier_generator = get_outlier_generator(
            self.data,
            self.time_features,
            self.scenario,
            min_deviation=self.min_deviation,
            max_deviation=self.max_deviation,
            p_outlier=self.p_outlier,
            start_interval=self.start_interval,
            end_interval=self.end_interval,
            seed=self.seed,
        )

        # 이상치 생성 (결측치가 아닌 영역에만)
        outlier_data, outlier_mask = outlier_generator.generate()
        self.outlier_mask = outlier_mask

        # 3. 결측치와 이상치가 겹치지 않도록 마스크 조정
        # 결측치가 있는 위치는 이상치로 표시하지 않음
        self.outlier_mask = self.outlier_mask & self.missing_mask

        # 4. 최종 데이터 생성
        # 결측치는 NaN으로, 이상치는 생성된 값으로 설정
        final_data = self.data.copy()
        final_data[~self.missing_mask] = np.nan  # 결측치 적용
        final_data[~self.outlier_mask] = outlier_data[~self.outlier_mask]  # 이상치 적용

        # 통계 정보 출력
        missing_ratio = 1.0 - np.mean(self.missing_mask)
        outlier_ratio = 1.0 - np.mean(self.outlier_mask)
        print("\n생성된 이상치/결측치 정보:")
        print(f"- 결측치 비율: {missing_ratio:.4f} ({missing_ratio * 100:.2f}%)")
        print(f"- 이상치 비율: {outlier_ratio:.4f} ({outlier_ratio * 100:.2f}%)")
        print(f"- 총 비정상 비율: {(missing_ratio + outlier_ratio):.4f} ({(missing_ratio + outlier_ratio) * 100:.2f}%)")

        return final_data, self.outlier_mask, self.missing_mask


def save_combined_data(data, outlier_mask, missing_mask, output_dir, scenario, **kwargs):
    """
    결측치와 이상치가 포함된 데이터를 저장

    Args:
        data: 결측치와 이상치가 포함된 데이터
        outlier_mask: 이상치 마스크
        missing_mask: 결측치 마스크
        output_dir: 출력 디렉토리
        scenario: 이상치 생성 시나리오
        **kwargs: 시나리오별 파라미터
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 폴더명에 시나리오와 파라미터 포함
    start_interval = kwargs.get("start_interval", 0.0)
    end_interval = kwargs.get("end_interval", 1.0)
    interval_str = f"_interval{start_interval:.2f}-{end_interval:.2f}"

    if scenario == "point":
        folder_name = (
            f"combined_{scenario}_dev{kwargs.get('min_deviation', 0.2):.2f}-"
            f"{kwargs.get('max_deviation', 0.5):.2f}_p{kwargs.get('p_outlier', 0.01):.4f}_"
            f"fault{kwargs.get('p_fault', 0.0015):.4f}_noise{kwargs.get('p_noise', 0.05):.4f}"
            f"{interval_str}"
        )
    elif scenario == "block":
        folder_name = (
            f"combined_{scenario}_dev{kwargs.get('min_deviation', 0.2):.2f}-"
            f"{kwargs.get('max_deviation', 0.5):.2f}_dur{kwargs.get('min_duration', 5)}-"
            f"{kwargs.get('max_duration', 20)}_p{kwargs.get('p_outlier', 0.005):.4f}_"
            f"fault{kwargs.get('p_fault', 0.0015):.4f}_noise{kwargs.get('p_noise', 0.05):.4f}"
            f"{interval_str}"
        )
    elif scenario == "contextual":
        folder_name = f"combined_{scenario}_ratio{kwargs.get('replace_ratio', 0.05):.2f}_" f"fault{kwargs.get('p_fault', 0.0015):.4f}_noise{kwargs.get('p_noise', 0.05):.4f}" f"{interval_str}"
    else:
        folder_name = f"combined_{scenario}{interval_str}"

    data_dir = output_path / folder_name
    data_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 저장
    np.save(os.path.join(data_dir, "data.npy"), data)
    np.save(os.path.join(data_dir, "outlier_mask.npy"), outlier_mask)
    np.save(os.path.join(data_dir, "missing_mask.npy"), missing_mask)

    # 필요한 보조 파일 복사
    for file in ["time_features_with_weather.txt", "node_subgraph.npy", "node_adjacent.txt", "node_dist.txt"]:
        src_file = os.path.join(STGAN_DATA_PATH, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(data_dir, file))
        else:
            print(f"경고: {file}을(를) 찾을 수 없습니다.")

    print(f"결측치와 이상치가 포함된 데이터가 {data_dir}에 저장되었습니다.")
    return data_dir


def create_combined_dataset(dataset_name, output_dir, scenario, **kwargs):
    """
    결측치와 이상치가 포함된 데이터셋 생성

    Args:
        dataset_name: 데이터셋 이름
        output_dir: 출력 디렉토리
        scenario: 이상치 생성 시나리오
        **kwargs: 시나리오별 파라미터
    """
    if dataset_name == "bay":
        data_path = os.path.join(STGAN_DATA_PATH, "data.npy")
        time_features_path = os.path.join(STGAN_DATA_PATH, "time_features_with_weather.txt")

        if not os.path.exists(data_path) or not os.path.exists(time_features_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path} 또는 {time_features_path}")

        # 데이터 로드
        print(f"데이터셋 '{dataset_name}' 로드 중...")
        data = np.load(data_path)
        time_features = np.loadtxt(time_features_path)
        print(f"데이터 형태: {data.shape}")
        print(f"시간 특성 형태: {time_features.shape}")

        # NaN 값 확인 및 처리
        if np.isnan(data).any():
            print("경고: 데이터에 NaN 값이 있습니다. 0으로 대체합니다.")
            data = np.nan_to_num(data, nan=0.0)

        # 결측치와 이상치 생성기 초기화
        generator = CombinedAnomalyGenerator(data, time_features, scenario, **kwargs)

        # 결측치와 이상치 생성
        start_interval = kwargs.get("start_interval", 0.0)
        end_interval = kwargs.get("end_interval", 1.0)
        print(f"'{scenario}' 시나리오로 결측치와 이상치 생성 중... (구간: {start_interval:.2f}-{end_interval:.2f})")
        combined_data, outlier_mask, missing_mask = generator.generate()

        # 결측치와 이상치가 포함된 데이터 저장
        data_dir = save_combined_data(combined_data, outlier_mask, missing_mask, output_dir, scenario, **kwargs)

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

        # 시나리오별 파라미터 설정
        generator_kwargs = {
            "seed": args.seed,
            "start_interval": args.start_interval,
            "end_interval": args.end_interval,
            "p_fault": args.p_fault,
            "p_noise": args.p_noise,
            "mask_type": args.mask_type,
        }

        if args.scenario == "point":
            generator_kwargs.update(
                {
                    "min_deviation": args.min_deviation,
                    "max_deviation": args.max_deviation,
                    "p_outlier": args.p_outlier,
                }
            )
        elif args.scenario == "block":
            generator_kwargs.update(
                {
                    "min_deviation": args.min_deviation,
                    "max_deviation": args.max_deviation,
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "p_outlier": args.p_outlier,
                }
            )
        elif args.scenario == "contextual":
            generator_kwargs.update(
                {
                    "replace_ratio": args.replace_ratio,
                }
            )

        # 결측치와 이상치가 포함된 데이터셋 생성
        data_dir = create_combined_dataset(args.dataset_name, output_dir, args.scenario, **generator_kwargs)

        print(f"\n모든 파일이 {data_dir} 디렉토리에 성공적으로 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGAN 데이터셋에 결측치와 이상치를 생성하는 스크립트")
    parser.add_argument("--dataset_name", type=str, default="bay", help="데이터셋 이름 (현재는 bay만 지원)")
    parser.add_argument("--output_dir", type=str, default="./datasets/bay/combined", help="출력 디렉토리 경로")
    parser.add_argument(
        "--scenario",
        type=str,
        default="point",
        choices=["point", "block", "contextual"],
        help="이상치 생성 시나리오",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--start_interval",
        type=float,
        default=0.0,
        help="이상치/결측치 생성 시작 구간 (0.0~1.0)",
    )
    parser.add_argument(
        "--end_interval",
        type=float,
        default=1.0,
        help="이상치/결측치 생성 종료 구간 (0.0~1.0)",
    )

    # 결측치 관련 파라미터
    parser.add_argument("--p_fault", type=float, default=0.1, help="결함 확률 (연속적인 결측치를 생성하는 비율)")
    parser.add_argument("--p_noise", type=float, default=0.1, help="잡음 확률 (독립적인 결측치를 생성하는 비율)")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="block",
        choices=["block", "point"],
        help="마스크 유형 (block: 연속적 결측치, point: 독립적 결측치)",
    )

    # 이상치 관련 파라미터
    parser.add_argument("--min_deviation", type=float, default=0.2, help="최소 편차 (정규화된 값)")
    parser.add_argument("--max_deviation", type=float, default=0.5, help="최대 편차 (정규화된 값)")
    parser.add_argument("--p_outlier", type=float, default=0.1, help="이상치 생성 확률")

    # 블록 이상치용 파라미터
    parser.add_argument("--min_duration", type=int, default=5, help="최소 지속 시간")
    parser.add_argument("--max_duration", type=int, default=20, help="최대 지속 시간")

    # 맥락 이상치용 파라미터
    parser.add_argument("--replace_ratio", type=float, default=0.05, help="대체 비율 (맥락 이상치)")

    args = parser.parse_args()
    main(args)
