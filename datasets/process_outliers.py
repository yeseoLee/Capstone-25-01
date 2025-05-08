"""
데이터셋에 이상치를 생성하고, 저장하는 스크립트

사용법:
    python ./datasets/process_outliers.py
    --dataset_name bay_block
    --output_dir ./datasets/processed/outliers
    --scenario point
"""

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# 데이터셋 경로 정의
datasets_path = {"bay": "./datasets/pems_bay", "la": "./datasets/metr_la"}


class OutlierGenerator:
    """
    이상치 생성기 기본 클래스
    """

    def __init__(self, df, seed=42):
        """
        초기화
        Args:
            df: 원본 데이터프레임
            seed: 랜덤 시드
        """
        self.df = df
        self.values = df.values
        self.outlier_mask = np.ones_like(self.values, dtype=bool)
        np.random.seed(seed)

    def generate(self):
        """
        이상치 생성 (상속 클래스에서 구현)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_outlier_mask(self):
        """
        이상치 마스크 반환
        """
        return self.outlier_mask


class PointOutlierGenerator(OutlierGenerator):
    """
    점 이상치 생성기
    """

    def __init__(self, df, seed=42, min_deviation=5, max_deviation=10, p_outlier=0.01):
        """
        초기화
        Args:
            df: 원본 데이터프레임
            seed: 랜덤 시드
            min_deviation: 최소 편차 (mph)
            max_deviation: 최대 편차 (mph)
            p_outlier: 이상치 생성 확률
        """
        super().__init__(df, seed)
        self.min_deviation = min_deviation
        self.max_deviation = max_deviation
        self.p_outlier = p_outlier

    def generate(self):
        """
        점 이상치 생성
        """
        # 각 센서별로 이상치 생성
        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[1]):
                if np.random.random() < self.p_outlier:
                    # 편차 방향 결정 (증가 또는 감소)
                    direction = np.random.choice([-1, 1])
                    # 편차 크기 결정
                    deviation = np.random.uniform(self.min_deviation, self.max_deviation)
                    # 이상치 생성
                    self.values[i, j] += direction * deviation
                    self.outlier_mask[i, j] = False


class CollectiveOutlierGenerator(OutlierGenerator):
    """
    집단적 이상치 생성기
    """

    def __init__(
        self, df, seed=42, min_deviation=5, max_deviation=10, min_duration=60, max_duration=240, p_outlier=0.005
    ):
        """
        초기화
        Args:
            df: 원본 데이터프레임
            seed: 랜덤 시드
            min_deviation: 최소 편차 (mph)
            max_deviation: 최대 편차 (mph)
            min_duration: 최소 지속 시간 (분)
            max_duration: 최대 지속 시간 (분)
            p_outlier: 이상치 생성 확률
        """
        super().__init__(df, seed)
        self.min_deviation = min_deviation
        self.max_deviation = max_deviation
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.p_outlier = p_outlier

    def generate(self):
        """
        집단적 이상치 생성
        """
        # 각 센서별로 이상치 생성
        for i in range(self.values.shape[0]):
            j = 0
            while j < self.values.shape[1]:
                if np.random.random() < self.p_outlier:
                    # 편차 방향 결정 (증가 또는 감소)
                    direction = np.random.choice([-1, 1])
                    # 편차 크기 결정
                    deviation = np.random.uniform(self.min_deviation, self.max_deviation)
                    # 지속 시간 결정 (분)
                    duration = np.random.randint(self.min_duration, self.max_duration + 1)
                    # 이상치 생성
                    end_idx = min(j + duration, self.values.shape[1])
                    self.values[i, j:end_idx] += direction * deviation
                    self.outlier_mask[i, j:end_idx] = False
                    j = end_idx
                else:
                    j += 1


class ContextualOutlierGenerator(OutlierGenerator):
    """
    맥락적/상황적 이상치 생성기
    """

    def __init__(self, df, seed=42):
        """
        초기화
        Args:
            df: 원본 데이터프레임
            seed: 랜덤 시드
        """
        super().__init__(df, seed)
        self.df = df
        self.index = df.index

    def generate(self):
        """
        맥락적/상황적 이상치 생성
        """
        # 시간대별 데이터 분류
        time_data = self._classify_by_time()

        # 새벽 시간대와 출근 시간대 교환
        self._swap_time_periods(time_data["dawn"], time_data["rush_hour"])

        # 주말과 평일 교환
        self._swap_weekday_weekend()

    def _classify_by_time(self):
        """
        시간대별로 데이터 분류
        """
        time_data = {
            "dawn": [],  # 새벽 2-3시
            "rush_hour": [],  # 출근 시간대 8-9시
            "weekday": [],  # 평일
            "weekend": [],  # 주말
        }

        for i, timestamp in enumerate(self.index):
            hour = timestamp.hour
            weekday = timestamp.weekday()

            if 2 <= hour < 3:
                time_data["dawn"].append(i)
            elif 8 <= hour < 9:
                time_data["rush_hour"].append(i)

            if weekday < 5:  # 평일 (0-4)
                time_data["weekday"].append(i)
            else:  # 주말 (5-6)
                time_data["weekend"].append(i)

        return time_data

    def _swap_time_periods(self, period1_indices, period2_indices):
        """
        두 시간대의 데이터 교환
        """
        if len(period1_indices) > 0 and len(period2_indices) > 0:
            # 랜덤하게 교환할 인덱스 선택
            n_swaps = min(len(period1_indices), len(period2_indices))
            p1_idx = np.random.choice(period1_indices, n_swaps, replace=False)
            p2_idx = np.random.choice(period2_indices, n_swaps, replace=False)

            # 데이터 교환
            for i in range(n_swaps):
                self.values[:, p1_idx[i]], self.values[:, p2_idx[i]] = (
                    self.values[:, p2_idx[i]].copy(),
                    self.values[:, p1_idx[i]].copy(),
                )
                self.outlier_mask[:, p1_idx[i]] = False
                self.outlier_mask[:, p2_idx[i]] = False

    def _swap_weekday_weekend(self):
        """
        평일과 주말의 동일 시간대 데이터 교환
        """
        time_data = self._classify_by_time()

        # 평일과 주말의 시간대별 데이터 교환
        for hour in range(24):
            weekday_indices = [i for i in time_data["weekday"] if self.index[i].hour == hour]
            weekend_indices = [i for i in time_data["weekend"] if self.index[i].hour == hour]

            self._swap_time_periods(weekday_indices, weekend_indices)


def create_outlier_dataset(original_df, outlier_mask, dataset_name):
    """
    원본 데이터프레임과 이상치 마스크를 사용하여 이상치가 있는 데이터셋을 생성합니다.

    Args:
        original_df: 원본 데이터프레임
        outlier_mask: 이상치 마스크 (True: 정상, False: 이상치)
        dataset_name: 데이터셋 이름

    Returns:
        outlier_df: 이상치가 포함된 데이터프레임
    """
    outlier_df = original_df.copy()
    outlier_values = outlier_df.values
    outlier_df = pd.DataFrame(outlier_values, index=original_df.index, columns=original_df.columns)
    return outlier_df


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


def get_outlier_generator(df, scenario, **kwargs):
    """
    시나리오에 따라 적절한 이상치 생성기를 반환합니다.

    Args:
        df: 원본 데이터프레임
        scenario: 이상치 생성 시나리오 ('point', 'collective', 'contextual')
        **kwargs: 추가 파라미터

    Returns:
        generator: 이상치 생성기 객체
    """
    if scenario == "point":
        return PointOutlierGenerator(df, **kwargs)
    elif scenario == "collective":
        return CollectiveOutlierGenerator(df, **kwargs)
    elif scenario == "contextual":
        return ContextualOutlierGenerator(df, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 시나리오: {scenario}")


def main(args):
    """
    메인 함수
    """
    try:
        # 데이터셋 로드
        print(f"데이터셋 '{args.dataset_name}' 로드 중...")

        # 데이터셋 파일 경로
        if "bay" in args.dataset_name:
            df = pd.read_hdf(os.path.join(datasets_path["bay"], "pems_bay.h5"))
        elif "la" in args.dataset_name:
            df = pd.read_hdf(os.path.join(datasets_path["la"], "metr_la.h5"))
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {args.dataset_name}")

        # 출력 디렉토리 생성
        output_base_dir = Path(args.output_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # 데이터셋별 출력 디렉토리 생성 - 시나리오와 파라미터 값을 폴더명에 포함
        folder_name = f"{args.dataset_name}_{args.scenario}"
        if args.scenario == "point":
            folder_name += f"_dev{args.min_deviation}-{args.max_deviation}_" f"p{args.p_outlier:.3f}"
        elif args.scenario == "collective":
            folder_name += (
                f"_dev{args.min_deviation}-{args.max_deviation}_"
                f"dur{args.min_duration}-{args.max_duration}_"
                f"p{args.p_outlier:.3f}"
            )

        dataset_output_dir = output_base_dir / folder_name
        dataset_output_dir.mkdir(exist_ok=True)

        # 이상치 생성기 초기화
        generator_kwargs = {
            "seed": args.seed,
            "min_deviation": args.min_deviation,
            "max_deviation": args.max_deviation,
            "p_outlier": args.p_outlier,
        }

        if args.scenario == "collective":
            generator_kwargs.update({"min_duration": args.min_duration, "max_duration": args.max_duration})

        generator = get_outlier_generator(df, args.scenario, **generator_kwargs)

        # 이상치 생성
        print(f"'{args.scenario}' 시나리오로 이상치 생성 중...")
        generator.generate()

        # 이상치가 포함된 데이터셋 생성
        outlier_df = create_outlier_dataset(df, generator.get_outlier_mask(), args.dataset_name)

        # 데이터셋 형식에 맞게 저장
        if "bay" in args.dataset_name:
            # PemsBay 데이터셋은 h5 형식으로 저장
            save_h5_dataset(outlier_df, os.path.join(dataset_output_dir, "pems_bay_outliers.h5"))
            # 보조 파일 복사
            copy_auxiliary_files(args.dataset_name, datasets_path["bay"], dataset_output_dir)

        elif "la" in args.dataset_name:
            # MetrLA 데이터셋은 h5 형식으로 저장
            save_h5_dataset(outlier_df, os.path.join(dataset_output_dir, "metr_la_outliers.h5"))
            # 보조 파일 복사
            copy_auxiliary_files(args.dataset_name, datasets_path["la"], dataset_output_dir)

        print(f"\n모든 파일이 {output_base_dir} 디렉토리에 성공적으로 저장되었습니다.")
        print(f"생성된 폴더: {folder_name}")
        print(f"사용된 시나리오: {args.scenario}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이상치가 있는 데이터셋을 생성하고 저장합니다.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bay_block",
        help="데이터셋 이름 (bay_block, la_block, bay_point, la_point)",
    )
    parser.add_argument("--output_dir", type=str, default="./datasets/outliers", help="출력 디렉토리 경로")
    parser.add_argument(
        "--scenario",
        type=str,
        default="point",
        choices=["point", "collective", "contextual"],
        help="이상치 생성 시나리오",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--min_deviation", type=float, default=5.0, help="최소 편차 (mph)")
    parser.add_argument("--max_deviation", type=float, default=10.0, help="최대 편차 (mph)")
    parser.add_argument("--p_outlier", type=float, default=0.01, help="이상치 생성 확률")
    parser.add_argument(
        "--min_duration", type=int, default=60, help="최소 지속 시간 (분, collective 시나리오에서만 사용)"
    )
    parser.add_argument(
        "--max_duration", type=int, default=240, help="최대 지속 시간 (분, collective 시나리오에서만 사용)"
    )

    args = parser.parse_args()
    main(args)
