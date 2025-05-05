import os

import numpy as np
import pandas as pd
import torch

from . import SpatioTemporalDataset, TemporalDataset


class ImputationDataset(TemporalDataset):
    def __init__(
        self,
        data,
        index=None,
        mask=None,
        eval_mask=None,
        freq=None,
        trend=None,
        scaler=None,
        window=24,
        stride=1,
        exogenous=None,
    ):
        if mask is None:
            mask = np.ones_like(data)
        if exogenous is None:
            exogenous = {}
        exogenous["mask_window"] = mask
        if eval_mask is not None:
            exogenous["eval_mask_window"] = eval_mask
        super(ImputationDataset, self).__init__(
            data,
            index=index,
            exogenous=exogenous,
            trend=trend,
            scaler=scaler,
            freq=freq,
            window=window,
            horizon=window,
            delay=-window,
            stride=stride,
        )

    def get(self, item, preprocess=False):
        res, transform = super(ImputationDataset, self).get(item, preprocess)
        res["x"] = torch.where(res["mask"], res["x"], torch.zeros_like(res["x"]))
        return res, transform

    def save_masked_data(self, save_path, file_prefix="masked_data"):  # noqa: C901
        """
        결측치가 있는 원본 데이터와 마스킹된 데이터를 함께 저장합니다.

        :param save_path: 저장할 디렉토리 경로
        :param file_prefix: 저장할 파일 이름의 접두사
        :return: 저장된 파일 경로 목록
        """
        # 디렉토리가 없으면 생성
        os.makedirs(save_path, exist_ok=True)

        # 데이터 준비
        original_data = self.data
        mask_data = self.mask_window if hasattr(self, "mask_window") else getattr(self, "mask", None)

        # 인덱스 준비
        index = self.index

        file_paths = []

        # 원본 데이터를 DataFrame으로 변환하여 저장
        if isinstance(original_data, torch.Tensor):
            original_data = original_data.cpu().numpy()

        # 다차원 데이터인 경우 (시간, 노드, 특성)
        if original_data.ndim > 2:
            # 각 노드별로 저장
            for node_idx in range(original_data.shape[1]):
                node_data = original_data[:, node_idx, :]

                # 노드 데이터를 DataFrame으로 변환
                if index is not None:
                    df_original = pd.DataFrame(node_data, index=index)
                else:
                    df_original = pd.DataFrame(node_data)

                # 마스크가 존재하는 경우
                if mask_data is not None:
                    mask_node = mask_data[:, node_idx, :] if mask_data.ndim > 2 else mask_data
                    if isinstance(mask_node, torch.Tensor):
                        mask_node = mask_node.cpu().numpy()

                    # 마스킹된 데이터 생성
                    masked_data = np.where(mask_node, node_data, np.nan)

                    if index is not None:
                        df_masked = pd.DataFrame(masked_data, index=index)
                    else:
                        df_masked = pd.DataFrame(masked_data)

                    # 마스크 자체도 저장
                    if index is not None:
                        df_mask = pd.DataFrame(mask_node, index=index)
                    else:
                        df_mask = pd.DataFrame(mask_node)

                    # 파일로 저장
                    original_path = os.path.join(save_path, f"{file_prefix}_node{node_idx}_original.csv")
                    masked_path = os.path.join(save_path, f"{file_prefix}_node{node_idx}_masked.csv")
                    mask_path = os.path.join(save_path, f"{file_prefix}_node{node_idx}_mask.csv")

                    df_original.to_csv(original_path)
                    df_masked.to_csv(masked_path)
                    df_mask.to_csv(mask_path)

                    file_paths.extend([original_path, masked_path, mask_path])
                else:
                    # 마스크가 없는 경우 원본 데이터만 저장
                    original_path = os.path.join(save_path, f"{file_prefix}_node{node_idx}_original.csv")
                    df_original.to_csv(original_path)
                    file_paths.append(original_path)
        else:
            # 단일 시계열 데이터인 경우
            if index is not None:
                df_original = pd.DataFrame(original_data, index=index)
            else:
                df_original = pd.DataFrame(original_data)

            # 마스크가 존재하는 경우
            if mask_data is not None:
                if isinstance(mask_data, torch.Tensor):
                    mask_data = mask_data.cpu().numpy()

                # 마스킹된 데이터 생성
                masked_data = np.where(mask_data, original_data, np.nan)

                if index is not None:
                    df_masked = pd.DataFrame(masked_data, index=index)
                    df_mask = pd.DataFrame(mask_data, index=index)
                else:
                    df_masked = pd.DataFrame(masked_data)
                    df_mask = pd.DataFrame(mask_data)

                # 파일로 저장
                original_path = os.path.join(save_path, f"{file_prefix}_original.csv")
                masked_path = os.path.join(save_path, f"{file_prefix}_masked.csv")
                mask_path = os.path.join(save_path, f"{file_prefix}_mask.csv")

                df_original.to_csv(original_path)
                df_masked.to_csv(masked_path)
                df_mask.to_csv(mask_path)

                file_paths.extend([original_path, masked_path, mask_path])
            else:
                # 마스크가 없는 경우 원본 데이터만 저장
                original_path = os.path.join(save_path, f"{file_prefix}_original.csv")
                df_original.to_csv(original_path)
                file_paths.append(original_path)

        return file_paths


class GraphImputationDataset(ImputationDataset, SpatioTemporalDataset):
    pass
