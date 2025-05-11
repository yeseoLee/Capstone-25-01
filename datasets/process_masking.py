"""
데이터셋에 결측치(마스킹)를 생성하고, 저장하는 스크립트

사용법:
    python ./datasets/process_masking.py --dataset_name bay_block --output_dir ./datasets/processed/masked
"""

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# 데이터셋 경로 정의
datasets_path = {"bay": "./datasets/pems_bay"}
stgan_path = "./datasets/stgan/pems_bay/data"


class ImputationDataset(Dataset):
    """
    결측치 보간을 위한 기본 데이터셋 클래스
    """

    def __init__(self, values, mask, eval_mask=None, idxs=None, window=24, stride=1):
        """
        초기화
        """
        self.values = values  # 원본 데이터
        self.mask = mask  # 훈련용 마스크
        self.eval_mask = eval_mask if eval_mask is not None else np.zeros_like(mask)  # 평가용 마스크
        self.window = window  # 윈도우 크기
        self.stride = stride  # 슬라이딩 윈도우 스트라이드
        self.idxs = np.arange(values.shape[0]) if idxs is None else idxs  # 인덱스

        self.horizon = 0
        self.compute_valid_indices()

    def compute_valid_indices(self):
        """
        유효한 인덱스 계산
        """
        self.valid_indices = []
        max_start = self.values.shape[1] - self.window - self.horizon + 1
        for i in range(0, max_start, self.stride):
            self.valid_indices.append(i)
        self.len = len(self.valid_indices) * self.values.shape[0]

    def __len__(self):
        """
        데이터셋 길이
        """
        return self.len

    def __getitem__(self, idx):
        """
        데이터셋에서 항목 가져오기
        """
        node_idx = idx // len(self.valid_indices)
        window_idx = self.valid_indices[idx % len(self.valid_indices)]

        values = self.values[node_idx, window_idx : window_idx + self.window + self.horizon]
        mask = self.mask[node_idx, window_idx : window_idx + self.window + self.horizon]
        eval_mask = self.eval_mask[node_idx, window_idx : window_idx + self.window + self.horizon]
        time_idx = np.arange(window_idx, window_idx + self.window + self.horizon)

        return {
            "values": torch.FloatTensor(values),
            "mask": torch.FloatTensor(mask),
            "eval_mask": torch.FloatTensor(eval_mask),
            "node_idx": self.idxs[node_idx],
            "time_idx": time_idx,
        }

    def save_masked_data(self, output_dir, file_prefix=None):
        """
        마스킹된 데이터를 CSV 형식으로 저장
        """
        if file_prefix is None:
            file_prefix = "masked_data"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 원본 데이터, 마스크, 결측치가 포함된 데이터 저장
        original_file = output_dir / f"{file_prefix}_original.csv"
        mask_file = output_dir / f"{file_prefix}_mask.csv"
        masked_file = output_dir / f"{file_prefix}_masked.csv"

        # 결측치가 포함된 데이터 생성
        masked_values = self.values.copy()
        masked_values[~self.mask.astype(bool)] = np.nan

        # CSV 파일 저장
        np.savetxt(original_file, self.values, delimiter=",")
        np.savetxt(mask_file, self.mask, delimiter=",")
        np.savetxt(masked_file, masked_values, delimiter=",")

        saved_files.extend([original_file, mask_file, masked_file])

        return saved_files


class GraphImputationDataset(ImputationDataset):
    """
    그래프 지원이 있는 결측치 보간 데이터셋
    """

    def __getitem__(self, idx):
        """
        데이터셋에서 항목 가져오기 (그래프 지원)
        """
        item = super().__getitem__(idx)

        # 그래프 정보 추가
        item["node_idx"] = torch.LongTensor([item["node_idx"]])

        return item


class MissingValuesBase:
    """
    결측치 기반 데이터셋의 기본 클래스
    """

    def __init__(self, p_fault=0.0015, p_noise=0.05, features=None):
        """
        초기화
        """
        self.p_fault = p_fault
        self.p_noise = p_noise
        self.features = features
        self.df = None
        self.training_mask = None
        self.eval_mask = None

    def dataframe(self):
        """
        데이터프레임 반환
        """
        return self.df

    def numpy(self, return_idx=False):
        """
        NumPy 배열 반환
        """
        values = self.df.values
        if return_idx:
            return values, self.df.index.values, self.df.columns.values
        return values

    def _create_missing_mask(self, df):
        """
        결측치 마스크 생성
        """
        values = df.values
        mask = np.ones_like(values)

        # 결함 (연속적인 결측치) 생성
        if self.p_fault > 0:
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if np.random.random() < self.p_fault:
                        # 결함 길이 (1~10 사이 임의 값)
                        fault_length = np.random.randint(1, 11)
                        if j + fault_length <= mask.shape[1]:
                            mask[i, j : j + fault_length] = 0

        # 잡음 (독립적인 결측치) 생성
        if self.p_noise > 0:
            noise_mask = np.random.random(mask.shape) < self.p_noise
            mask[noise_mask] = 0

        return mask


class MissingValuesPemsBay(MissingValuesBase):
    """
    PemsBay 데이터셋에 결측치를 적용한 클래스
    """

    def __init__(self, p_fault=0.0015, p_noise=0.05, features=None):
        """
        초기화 및 데이터 로드
        """
        super().__init__(p_fault, p_noise, features)

        # STGAN 형식의 PemsBay 데이터 로드
        try:
            # STGAN 형식으로 저장된 데이터 로드 시도
            stgan_data = np.load(os.path.join(stgan_path, "data.npy"))
            # (num_timestamps, num_nodes, 1, 2) -> (num_timestamps, num_nodes) 형태로 변환
            data = stgan_data[:, :, 0, 0]  # 첫 번째 특성(속도)만 사용

            # DataFrame으로 변환
            self.df = pd.DataFrame(data)

        except (FileNotFoundError, ValueError):
            # 기존 방식으로 데이터 로드
            self.df = pd.read_hdf(os.path.join(datasets_path["bay"], "pems_bay.h5"))

            if self.features is not None:
                self.df = self.df[self.features]

        self.training_mask = self._create_missing_mask(self.df).astype(bool)
        # 비트 연산자 대신 논리 연산자 사용
        mask2 = self._create_missing_mask(self.df).astype(bool)
        self.eval_mask = mask2 & (~self.training_mask)


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


def save_stgan_format(masked_values, output_dir):
    """
    마스킹된 데이터를 STGAN 형식으로 저장합니다.

    Args:
        masked_values: 마스킹된 데이터값 (2D 배열)
        output_dir: 출력 디렉토리 경로
    """
    try:
        # 원본 STGAN 데이터 로드
        original_data = np.load(os.path.join(stgan_path, "data.npy"))

        # 마스킹된 데이터로 STGAN 형식 데이터 생성
        masked_data = original_data.copy()

        # 마스킹된 2D 데이터를 4D로 변환
        for i in range(masked_values.shape[0]):
            for j in range(masked_values.shape[1]):
                if np.isnan(masked_values[i, j]):
                    # 결측치는 모든 특성에 적용
                    masked_data[i, j, :, :] = np.nan
                else:
                    # 값 복사
                    masked_data[i, j, 0, 0] = masked_values[i, j]
                    masked_data[i, j, 0, 1] = masked_values[i, j]  # 두 번째 특성도 동일한 값 사용

        # STGAN 형식으로 저장
        np.save(os.path.join(output_dir, "data.npy"), masked_data)

        # 필요한 보조 파일 복사
        for file in ["time_features.txt", "node_subgraph.npy", "node_adjacent.txt", "node_dist.txt"]:
            src_file = os.path.join(stgan_path, file)
            dst_file = os.path.join(output_dir, file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)

        print(f"STGAN 형식 데이터가 {output_dir}에 저장되었습니다.")

    except FileNotFoundError:
        print("원본 STGAN 형식 데이터를 찾을 수 없습니다. 먼저 prepare_datasets.py를 실행하세요.")


def copy_auxiliary_files(dataset_name, original_dir, output_dir):
    """
    필요한 보조 파일(예: 거리 행렬, 센서 ID 등)을 복사합니다.

    Args:
        dataset_name: 데이터셋 이름
        original_dir: 원본 데이터 디렉토리
        output_dir: 출력 디렉토리
    """
    # PemsBay 데이터셋의 보조 파일 복사
    files_to_copy = ["pems_bay_dist.npy", "distances_bay.csv"]
    for file in files_to_copy:
        if os.path.exists(os.path.join(original_dir, file)):
            shutil.copy2(os.path.join(original_dir, file), os.path.join(output_dir, file))
            print(f"보조 파일 {file}이 복사되었습니다.")


def get_dataset(dataset_name, p_fault=0.0015, p_noise=0.05):
    """
    데이터셋 이름에 따라 적절한 데이터셋 객체를 반환합니다.

    Args:
        dataset_name: 데이터셋 이름
        p_fault: 결함 확률
        p_noise: 잡음 확률

    Returns:
        dataset: 데이터셋 객체
    """
    if dataset_name == "bay_block":
        dataset = MissingValuesPemsBay(p_fault=p_fault, p_noise=p_noise)
    elif dataset_name == "bay_point":
        dataset = MissingValuesPemsBay(p_fault=0.0, p_noise=0.25)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def main(args):
    """
    메인 함수
    """
    try:
        # 데이터셋 로드
        print(f"데이터셋 '{args.dataset_name}' 로드 중...")
        dataset = get_dataset(args.dataset_name, args.p_fault, args.p_noise)

        # 데이터셋 정보 확인
        original_df = dataset.dataframe()
        # mask = dataset.mask
        # eval_mask = dataset.eval_mask

        # 출력 디렉토리 생성
        output_base_dir = Path(args.output_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # 데이터셋별 출력 디렉토리 생성 - 파라미터 값을 폴더명에 포함
        folder_name = (
            f"{args.dataset_name}_w{args.window}_s{args.stride}_" f"fault{args.p_fault:.4f}_noise{args.p_noise:.4f}"
        )
        dataset_output_dir = output_base_dir / folder_name
        dataset_output_dir.mkdir(exist_ok=True)

        # STGAN 형식 출력 디렉토리 생성
        stgan_output_dir = output_base_dir / (folder_name + "_stgan")
        stgan_output_dir.mkdir(exist_ok=True)

        # 결측치가 있는 데이터셋 생성
        masked_df = create_masked_dataset(original_df, dataset.training_mask, args.dataset_name)

        # PemsBay 데이터셋은 h5 형식으로 저장
        save_h5_dataset(masked_df, os.path.join(dataset_output_dir, "pems_bay_masked.h5"))
        # 보조 파일 복사
        copy_auxiliary_files(args.dataset_name, datasets_path["bay"], dataset_output_dir)

        # STGAN 형식으로 저장
        save_stgan_format(masked_df.values, stgan_output_dir)

        # CSV 형식으로도 저장 (선택 사항)
        if args.save_csv:
            # CSV 폴더명에도 파라미터 값 포함
            csv_folder_name = (
                f"{args.dataset_name}_w{args.window}_s{args.stride}_"
                f"fault{args.p_fault:.4f}_noise{args.p_noise:.4f}_csv"
            )
            csv_output_dir = output_base_dir / csv_folder_name
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
        print(f"생성된 폴더: {folder_name}")
        print(f"STGAN 형식 데이터 폴더: {folder_name}_stgan")
        print(f"사용된 마스크: 결함 확률={args.p_fault}, 잡음 확률={args.p_noise}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="결측치가 있는 원본 데이터와 마스킹된 데이터를 함께 저장합니다.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bay_block",
        help="데이터셋 이름 (bay_block, bay_point)",
    )
    parser.add_argument("--output_dir", type=str, default="./datasets/masked", help="출력 디렉토리 경로")
    parser.add_argument("--window", type=int, default=24, help="윈도우 크기")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 스트라이드")
    parser.add_argument("--p_fault", type=float, default=0.0015, help="결함 확률 (연속적인 결측치를 생성하는 비율)")
    parser.add_argument("--p_noise", type=float, default=0.05, help="잡음 확률 (독립적인 결측치를 생성하는 비율)")
    parser.add_argument("--save_csv", type=bool, default=False, help="CSV 파일도 함께 저장할지 여부")

    args = parser.parse_args()
    main(args)
