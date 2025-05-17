"""
STGAN 데이터셋에 이상치(outlier)를 생성하고 저장하는 스크립트

사용법:
    python ./STGAN/data_pipeline/create_outliers.py --dataset_name bay --output_dir ./STGAN/bay/outliers --scenario point
"""  # noqa: E501

import argparse
import os
from pathlib import Path
import shutil

import numpy as np


# STGAN 데이터 경로 정의
STGAN_DATA_PATH = "./STGAN/bay/data"


class OutlierGenerator:
    """
    이상치 생성기 기본 클래스
    """

    def __init__(self, data, time_features, seed=42, start_interval=0.0, end_interval=1.0):
        """
        초기화
        Args:
            data: 원본 데이터 (4D 배열: 시간, 노드, 특성, 채널)
            time_features: 시간 특성 데이터 (2D 배열: 시간, 특성[요일+시간])
            seed: 랜덤 시드
            start_interval: 이상치 생성 시작 구간 (0.0~1.0)
            end_interval: 이상치 생성 종료 구간 (0.0~1.0)
        """
        self.data = data.copy()
        self.time_features = time_features
        self.outlier_mask = np.ones_like(data, dtype=bool)  # True: 정상, False: 이상치
        np.random.seed(seed)

        # 시작 및 종료 구간 계산
        time_len = self.data.shape[0]
        self.start_idx = int(time_len * start_interval)
        self.end_idx = int(time_len * end_interval)
        self.start_interval = start_interval
        self.end_interval = end_interval

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

    def get_data_with_outliers(self):
        """
        이상치가 포함된 데이터 반환
        """
        return self.data


class PointOutlierGenerator(OutlierGenerator):
    """
    점 이상치 생성기 - 임의의 데이터 포인트에 이상치 추가
    """

    def __init__(
        self,
        data,
        time_features,
        seed=42,
        min_deviation=0.2,
        max_deviation=0.5,
        p_outlier=0.01,
        start_interval=0.0,
        end_interval=1.0,
    ):
        """
        초기화
        Args:
            data: 원본 데이터
            time_features: 시간 특성 데이터
            seed: 랜덤 시드
            min_deviation: 최소 편차 (정규화된 값)
            max_deviation: 최대 편차 (정규화된 값)
            p_outlier: 이상치 생성 확률
            start_interval: 이상치 생성 시작 구간 (0.0~1.0)
            end_interval: 이상치 생성 종료 구간 (0.0~1.0)
        """
        super().__init__(data, time_features, seed, start_interval, end_interval)
        self.min_deviation = min_deviation
        self.max_deviation = max_deviation
        self.p_outlier = p_outlier

    def generate(self):
        """
        점 이상치 생성
        """
        print(
            f"점 이상치 생성 중... (편차 범위: {self.min_deviation:.2f}-{self.max_deviation:.2f}, "
            f"확률: {self.p_outlier:.4f}, 구간: {self.start_interval:.2f}-{self.end_interval:.2f})"
        )

        # 데이터 형태 확인
        time_len, node_len, feature_dim, channel_dim = self.data.shape

        # 이상치 생성할 위치 결정 (시간, 노드)
        outlier_mask = np.random.random((time_len, node_len)) < self.p_outlier

        # 이상치 생성 (지정된 구간만)
        for t in range(self.start_idx, self.end_idx):
            for n in range(node_len):
                if outlier_mask[t, n]:
                    # 편차 방향 결정 (증가 또는 감소)
                    direction = np.random.choice([-1, 1])
                    # 편차 크기 결정
                    deviation = np.random.uniform(self.min_deviation, self.max_deviation)

                    # 모든 특성에 이상치 적용
                    for f in range(feature_dim):
                        for c in range(channel_dim):
                            self.data[t, n, f, c] += direction * deviation
                            self.outlier_mask[t, n, f, c] = False

        # 이상치 개수 계산
        outlier_count = np.sum(~self.outlier_mask) // (feature_dim * channel_dim)
        print(
            f"생성된 점 이상치 개수: {outlier_count} / {time_len * node_len} ({outlier_count / (time_len * node_len) * 100:.2f}%)"  # noqa: E501
        )

        return self.data, self.outlier_mask


class BlockOutlierGenerator(OutlierGenerator):
    """
    블록 이상치 생성기 - 연속된 시간대에 걸쳐 이상치 추가
    """

    def __init__(
        self,
        data,
        time_features,
        seed=42,
        min_deviation=0.2,
        max_deviation=0.5,
        min_duration=5,
        max_duration=20,
        p_outlier=0.005,
        start_interval=0.0,
        end_interval=1.0,
    ):
        """
        초기화
        Args:
            data: 원본 데이터
            time_features: 시간 특성 데이터
            seed: 랜덤 시드
            min_deviation: 최소 편차 (정규화된 값)
            max_deviation: 최대 편차 (정규화된 값)
            min_duration: 최소 지속 시간 (시간 단위)
            max_duration: 최대 지속 시간 (시간 단위)
            p_outlier: 이상치 발생 확률
            start_interval: 이상치 생성 시작 구간 (0.0~1.0)
            end_interval: 이상치 생성 종료 구간 (0.0~1.0)
        """
        super().__init__(data, time_features, seed, start_interval, end_interval)
        self.min_deviation = min_deviation
        self.max_deviation = max_deviation
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.p_outlier = p_outlier

    def generate(self):
        """
        블록 이상치 생성
        """
        print(
            f"블록 이상치 생성 중... (편차 범위: {self.min_deviation:.2f}-{self.max_deviation:.2f}, "
            + f"지속 시간: {self.min_duration}-{self.max_duration}, 확률: {self.p_outlier:.4f}, "
            + f"구간: {self.start_interval:.2f}-{self.end_interval:.2f})"
        )

        # 데이터 형태 확인
        time_len, node_len, feature_dim, channel_dim = self.data.shape

        # 각 노드별로 이상치 블록 생성
        for n in range(node_len):
            t = self.start_idx
            while t < self.end_idx:
                if np.random.random() < self.p_outlier:
                    # 편차 방향 결정 (증가 또는 감소)
                    direction = np.random.choice([-1, 1])
                    # 편차 크기 결정
                    deviation = np.random.uniform(self.min_deviation, self.max_deviation)
                    # 지속 시간 결정
                    duration = np.random.randint(self.min_duration, self.max_duration + 1)

                    # 블록 이상치 적용
                    end_t = min(t + duration, self.end_idx)
                    for i in range(t, end_t):
                        for f in range(feature_dim):
                            for c in range(channel_dim):
                                self.data[i, n, f, c] += direction * deviation
                                self.outlier_mask[i, n, f, c] = False

                    t = end_t
                else:
                    t += 1

        # 이상치 개수 계산
        outlier_count = np.sum(~self.outlier_mask) // (feature_dim * channel_dim)
        print(
            f"생성된 블록 이상치 개수: {outlier_count} / {time_len * node_len} ({outlier_count / (time_len * node_len) * 100:.2f}%)"  # noqa: E501
        )

        return self.data, self.outlier_mask


class ContextualOutlierGenerator(OutlierGenerator):
    """
    맥락적 이상치 생성기 - 특정 맥락에서 부적절한 데이터 패턴 생성
    """

    def __init__(self, data, time_features, seed=42, replace_ratio=0.05, start_interval=0.0, end_interval=1.0):
        """
        초기화
        Args:
            data: 원본 데이터
            time_features: 시간 특성 데이터 (첫 7개: 요일, 다음 24개: 시간)
            seed: 랜덤 시드
            replace_ratio: 대체 비율 (맥락 이상치를 생성할 데이터의 비율)
            start_interval: 이상치 생성 시작 구간 (0.0~1.0)
            end_interval: 이상치 생성 종료 구간 (0.0~1.0)
        """
        super().__init__(data, time_features, seed, start_interval, end_interval)
        self.replace_ratio = replace_ratio

    def generate(self):
        """
        맥락적 이상치 생성
        """
        print(
            f"맥락적 이상치 생성 중... (대체 비율: {self.replace_ratio:.2f}, 구간: {self.start_interval:.2f}-{self.end_interval:.2f})"  # noqa: E501
        )

        # 시간 특성 분석
        time_len = self.time_features.shape[0]

        # 요일 및 시간 인덱스 추출
        day_indices = self.get_day_indices()
        hour_indices = self.get_hour_indices()

        # 출퇴근 시간과 새벽 시간 교환
        self.swap_rush_dawn_hours(day_indices, hour_indices)

        # 주말과 평일 교환
        self.swap_weekday_weekend(day_indices, hour_indices)

        # 이상치 개수 계산
        outlier_count = np.sum(~self.outlier_mask) // (self.data.shape[2] * self.data.shape[3])
        print(
            f"생성된 맥락적 이상치 개수: {outlier_count} / {time_len * self.data.shape[1]} ({outlier_count / (time_len * self.data.shape[1]) * 100:.2f}%)"  # noqa: E501
        )

        return self.data, self.outlier_mask

    def get_day_indices(self):
        """
        요일별 시간 인덱스 추출
        """
        day_indices = {}

        # 요일 정보는 첫 7개 열에 원-핫 인코딩
        for day in range(7):  # 0: 월요일, ..., 6: 일요일
            indices = np.where(self.time_features[:, day] == 1)[0]
            # 지정된 구간 내 인덱스만 필터링
            day_indices[day] = [idx for idx in indices if self.start_idx <= idx < self.end_idx]

        return day_indices

    def get_hour_indices(self):
        """
        시간대별 시간 인덱스 추출
        """
        hour_indices = {}

        # 시간 정보는 7~30 열에 원-핫 인코딩
        for hour in range(24):
            indices = np.where(self.time_features[:, hour + 7] == 1)[0]
            # 지정된 구간 내 인덱스만 필터링
            hour_indices[hour] = [idx for idx in indices if self.start_idx <= idx < self.end_idx]

        return hour_indices

    def swap_rush_dawn_hours(self, day_indices, hour_indices):
        """
        출퇴근 시간과 새벽 시간 교환
        """
        # 출퇴근 시간대 (8-9시, 17-18시)
        rush_hours = [8, 9, 17, 18]
        # 새벽 시간대 (2-4시)
        dawn_hours = [2, 3, 4]

        # 평일(0-4) 기간 동안만 교환
        weekday_indices = []
        for day in range(0, 5):  # 월~금
            if day in day_indices:
                weekday_indices.extend(day_indices[day])

        # 출퇴근 시간 인덱스
        rush_indices = []
        for hour in rush_hours:
            rush_indices.extend(set(hour_indices[hour]).intersection(weekday_indices))

        # 새벽 시간 인덱스
        dawn_indices = []
        for hour in dawn_hours:
            dawn_indices.extend(set(hour_indices[hour]).intersection(weekday_indices))

        # 교환할 데이터 포인트 수 결정
        swap_count = int(min(len(rush_indices), len(dawn_indices)) * self.replace_ratio)

        if swap_count > 0:
            # 무작위로 교환할 시간 인덱스 선택
            rush_swap = np.random.choice(rush_indices, swap_count, replace=False)
            dawn_swap = np.random.choice(dawn_indices, swap_count, replace=False)

            # 데이터 교환
            self.swap_data_at_indices(rush_swap, dawn_swap)

            print(f"출퇴근 시간과 새벽 시간 교환: {swap_count}개")

    def swap_weekday_weekend(self, day_indices, hour_indices):
        """
        주말과 평일 교환
        """
        # 평일 인덱스 (0-4)
        weekday_indices = []
        for day in range(0, 5):  # 월~금
            if day in day_indices:
                weekday_indices.extend(day_indices[day])

        # 주말 인덱스 (5-6)
        weekend_indices = []
        for day in range(5, 7):  # 토~일
            if day in day_indices:
                weekend_indices.extend(day_indices[day])

        # 낮 시간대 (10-16시)
        daytime_hours = list(range(10, 17))

        # 평일 낮 시간 인덱스
        weekday_daytime = []
        for hour in daytime_hours:
            weekday_daytime.extend(set(hour_indices[hour]).intersection(weekday_indices))

        # 주말 낮 시간 인덱스
        weekend_daytime = []
        for hour in daytime_hours:
            weekend_daytime.extend(set(hour_indices[hour]).intersection(weekend_indices))

        # 교환할 데이터 포인트 수 결정
        swap_count = int(min(len(weekday_daytime), len(weekend_daytime)) * self.replace_ratio)

        if swap_count > 0:
            # 무작위로 교환할 시간 인덱스 선택
            weekday_swap = np.random.choice(weekday_daytime, swap_count, replace=False)
            weekend_swap = np.random.choice(weekend_daytime, swap_count, replace=False)

            # 데이터 교환
            self.swap_data_at_indices(weekday_swap, weekend_swap)

            print(f"평일 낮 시간과 주말 낮 시간 교환: {swap_count}개")

    def swap_data_at_indices(self, indices1, indices2):
        """
        두 인덱스 집합 간 데이터 교환
        """
        for i in range(len(indices1)):
            # 데이터 교환
            idx1, idx2 = indices1[i], indices2[i]
            temp = self.data[idx1].copy()
            self.data[idx1] = self.data[idx2].copy()
            self.data[idx2] = temp

            # 마스크 업데이트 (두 곳 모두 이상치로 표시)
            self.outlier_mask[idx1] = False
            self.outlier_mask[idx2] = False


def get_outlier_generator(data, time_features, scenario, **kwargs):
    """
    시나리오에 따라 적절한 이상치 생성기를 반환합니다.

    Args:
        data: 원본 데이터
        time_features: 시간 특성 데이터
        scenario: 이상치 생성 시나리오 ('point', 'block', 'contextual')
        **kwargs: 추가 파라미터

    Returns:
        generator: 이상치 생성기 객체
    """
    if scenario == "point":
        return PointOutlierGenerator(data, time_features, **kwargs)
    elif scenario == "block":
        return BlockOutlierGenerator(data, time_features, **kwargs)
    elif scenario == "contextual":
        return ContextualOutlierGenerator(data, time_features, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 시나리오: {scenario}")


def save_outlier_data(data, outlier_mask, output_dir, scenario, **kwargs):
    """
    이상치가 포함된 데이터를 저장

    Args:
        data: 이상치가 포함된 데이터
        outlier_mask: 이상치 마스크 (True: 정상, False: 이상치)
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
            f"{scenario}_dev{kwargs.get('min_deviation', 0.2):.2f}-"
            f"{kwargs.get('max_deviation', 0.5):.2f}_"
            f"p{kwargs.get('p_outlier', 0.01):.4f}"
            f"{interval_str}"
        )
    elif scenario == "block":
        folder_name = (
            f"{scenario}_dev{kwargs.get('min_deviation', 0.2):.2f}-"
            f"{kwargs.get('max_deviation', 0.5):.2f}_"
            f"dur{kwargs.get('min_duration', 5)}-{kwargs.get('max_duration', 20)}_"
            f"p{kwargs.get('p_outlier', 0.005):.4f}"
            f"{interval_str}"
        )
    elif scenario == "contextual":
        folder_name = f"{scenario}_ratio{kwargs.get('replace_ratio', 0.05):.2f}{interval_str}"
    else:
        folder_name = f"{scenario}{interval_str}"

    data_dir = output_path / folder_name
    data_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 저장
    np.save(os.path.join(data_dir, "data.npy"), data)
    np.save(os.path.join(data_dir, "outlier_mask.npy"), outlier_mask)  # 이상치 마스크도 저장

    # 필요한 보조 파일 복사
    for file in ["time_features.txt", "node_subgraph.npy", "node_adjacent.txt", "node_dist.txt"]:
        src_file = os.path.join(STGAN_DATA_PATH, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(data_dir, file))

    print(f"이상치가 포함된 데이터가 {data_dir}에 저장되었습니다.")
    return data_dir


def create_outlier_dataset(dataset_name, output_dir, scenario, **kwargs):
    """
    이상치가 포함된 데이터셋 생성

    Args:
        dataset_name: 데이터셋 이름
        output_dir: 출력 디렉토리
        scenario: 이상치 생성 시나리오
        **kwargs: 시나리오별 파라미터
    """
    if dataset_name == "bay":
        data_path = os.path.join(STGAN_DATA_PATH, "data.npy")
        time_features_path = os.path.join(STGAN_DATA_PATH, "time_features.txt")

        if not os.path.exists(data_path) or not os.path.exists(time_features_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path} 또는 {time_features_path}")

        # 데이터 로드
        print(f"데이터셋 '{dataset_name}' 로드 중...")
        data = np.load(data_path)
        time_features = np.loadtxt(time_features_path)
        print(f"데이터 형태: {data.shape}")
        print(f"시간 특성 형태: {time_features.shape}")

        # 이상치 생성기 가져오기
        generator = get_outlier_generator(data, time_features, scenario, **kwargs)

        # 이상치 생성
        start_interval = kwargs.get("start_interval", 0.0)
        end_interval = kwargs.get("end_interval", 1.0)
        print(f"'{scenario}' 시나리오로 이상치 생성 중... (구간: {start_interval:.2f}-{end_interval:.2f})")
        outlier_data, outlier_mask = generator.generate()

        # 이상치가 포함된 데이터 저장
        data_dir = save_outlier_data(outlier_data, outlier_mask, output_dir, scenario, **kwargs)

        # 이상치 비율 계산
        outlier_ratio = 1.0 - np.mean(outlier_mask)
        print("생성된 이상치 정보:")
        print(f"- 이상치 비율: {outlier_ratio:.4f} ({outlier_ratio*100:.2f}%)")
        print(f"- 이상치 마스크 형태: {outlier_mask.shape}")

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
        generator_kwargs = {"seed": args.seed, "start_interval": args.start_interval, "end_interval": args.end_interval}

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

        # 이상치가 포함된 데이터셋 생성
        data_dir = create_outlier_dataset(args.dataset_name, output_dir, args.scenario, **generator_kwargs)

        print(f"\n모든 파일이 {data_dir} 디렉토리에 성공적으로 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGAN 데이터셋에 이상치를 생성하는 스크립트")
    parser.add_argument("--dataset_name", type=str, default="bay", help="데이터셋 이름 (현재는 bay만 지원)")
    parser.add_argument("--output_dir", type=str, default="./STGAN/bay/outliers", help="출력 디렉토리 경로")
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
        help="이상치 생성 시작 구간 (0.0~1.0)",
    )
    parser.add_argument(
        "--end_interval",
        type=float,
        default=1.0,
        help="이상치 생성 종료 구간 (0.0~1.0)",
    )

    # 점/블록 이상치용 파라미터
    parser.add_argument("--min_deviation", type=float, default=0.2, help="최소 편차 (정규화된 값)")
    parser.add_argument("--max_deviation", type=float, default=0.5, help="최대 편차 (정규화된 값)")
    parser.add_argument("--p_outlier", type=float, default=0.01, help="이상치 생성 확률")

    # 블록 이상치용 파라미터
    parser.add_argument("--min_duration", type=int, default=5, help="최소 지속 시간")
    parser.add_argument("--max_duration", type=int, default=20, help="최대 지속 시간")

    # 맥락 이상치용 파라미터
    parser.add_argument("--replace_ratio", type=float, default=0.05, help="대체 비율 (맥락 이상치)")

    args = parser.parse_args()
    main(args)
