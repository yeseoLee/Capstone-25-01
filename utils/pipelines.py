import copy

import numpy as np
import torch


class ImputationPipeline:
    """임퓨테이션 파이프라인 기본 클래스"""

    def __init__(self, missing_model, outlier_model, device="cuda"):
        """
        초기화
        Args:
            missing_model: 결측치 보간 모델
            outlier_model: 이상치 보간 모델
            device: 연산 디바이스
        """
        self.missing_model = missing_model
        self.outlier_model = outlier_model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.name = "Base Pipeline"

    def impute(self, data_dict):
        """
        데이터 보간 (상속 클래스에서 구현)
        Args:
            data_dict: 데이터 딕셔너리
        Returns:
            imputed_data: 보간된 데이터
        """
        raise NotImplementedError("Subclasses must implement this method")


class MissingFirstPipeline(ImputationPipeline):
    """결측치 보간 후 이상치 보간하는 파이프라인"""

    def __init__(self, missing_model, outlier_model, device="cuda"):
        super().__init__(missing_model, outlier_model, device)
        self.name = "결측치 → 이상치 보간 파이프라인"

    def impute(self, data_dict):
        """
        결측치를 먼저 보간한 후 이상치를 보간
        이 방식에서는 결측치 보간 시 이상치를 정상값으로 간주
        Args:
            data_dict: 데이터 딕셔너리
        Returns:
            imputed_data: 보간된 데이터
        """
        # 원본 데이터 복사
        data_copy = copy.deepcopy(data_dict)

        # 1단계: 결측치 보간 (이상치는 정상값으로 간주)
        missing_imputed = self.missing_model.impute(data_copy, device=self.device)

        # 2단계: 결측치 보간 결과를 기반으로 이상치 보간
        data_copy["mixed_data"] = missing_imputed
        final_imputed = self.outlier_model.impute(data_copy, device=self.device)

        return final_imputed


class TempFillPipeline(ImputationPipeline):
    """임시 대체 → 이상치 보간 → 결측치 보간 파이프라인"""

    def __init__(self, missing_model, outlier_model, device="cuda", temp_value_type="zero"):
        """
        초기화
        Args:
            missing_model: 결측치 보간 모델
            outlier_model: 이상치 보간 모델
            device: 연산 디바이스
            temp_value_type: 임시 대체 값 유형 ('zero', 'mean', 'neighbor')
        """
        super().__init__(missing_model, outlier_model, device)
        self.temp_value_type = temp_value_type
        self.name = f"임시 대체({temp_value_type}) → 이상치 → 결측치 보간 파이프라인"

    def _get_temp_values(self, data, missing_mask):
        """
        임시 대체 값 계산
        Args:
            data: 원본 데이터
            missing_mask: 결측치 마스크
        Returns:
            temp_values: 임시 대체 값
        """
        if self.temp_value_type == "zero":
            return self._get_zero_values(data)
        elif self.temp_value_type == "mean":
            return self._get_mean_values(data, missing_mask)
        elif self.temp_value_type == "neighbor":
            return self._get_neighbor_values(data, missing_mask)
        else:
            raise ValueError(f"지원하지 않는 임시 대체 값 유형: {self.temp_value_type}")

    def _get_zero_values(self, data):
        """영(0) 값으로 대체"""
        return np.zeros_like(data)

    def _get_mean_values(self, data, missing_mask):
        """각 노드의 평균값으로 대체"""
        # 각 노드별 평균 계산 (결측치 제외)
        node_means = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            valid_data = data[i, ~missing_mask[i]]
            node_means[i] = valid_data.mean() if len(valid_data) > 0 else 0

        # 평균값으로 대체
        return np.tile(node_means.reshape(-1, 1), (1, data.shape[1]))

    def _get_neighbor_values(self, data, missing_mask):
        """이웃 값의 평균으로 대체"""
        temp_values = np.copy(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if missing_mask[i, j]:
                    temp_values[i, j] = self._calculate_neighbor_value(data, missing_mask, i, j)
        return temp_values

    def _calculate_neighbor_value(self, data, missing_mask, i, j):
        """특정 위치의 이웃 값 계산"""
        neighbors = []
        # 이전 값
        if j > 0 and not missing_mask[i, j - 1]:
            neighbors.append(data[i, j - 1])
        # 다음 값
        if j < data.shape[1] - 1 and not missing_mask[i, j + 1]:
            neighbors.append(data[i, j + 1])

        if neighbors:
            return np.mean(neighbors)

        # 이웃 값이 없으면 해당 노드의 평균 사용
        valid_data = data[i, ~missing_mask[i]]
        return valid_data.mean() if len(valid_data) > 0 else 0

    def impute(self, data_dict):
        """
        임시 대체 값으로 결측치를 채운 후 이상치 보간, 이후 결측치 보간
        Args:
            data_dict: 데이터 딕셔너리
        Returns:
            imputed_data: 보간된 데이터
        """
        # 원본 데이터 복사
        data_copy = copy.deepcopy(data_dict)
        mixed_data = data_copy["mixed_data"]
        missing_mask = data_copy["missing_mask"]

        # 1단계: 결측치를 임시 값으로 대체
        temp_values = self._get_temp_values(mixed_data, missing_mask)
        temp_filled_data = np.copy(mixed_data)
        temp_filled_data[missing_mask] = temp_values[missing_mask]

        # 임시 대체된 데이터로 업데이트
        data_copy["mixed_data"] = temp_filled_data

        # 2단계: 이상치 보간
        outlier_imputed = self.outlier_model.impute(data_copy, device=self.device)

        # 3단계: 임시 대체된 결측치 다시 원래대로 (0 또는 NaN으로)
        outlier_imputed[missing_mask] = 0

        # 4단계: 결측치 보간
        data_copy["mixed_data"] = outlier_imputed
        final_imputed = self.missing_model.impute(data_copy, device=self.device)

        return final_imputed


class AlternatingPipeline(ImputationPipeline):
    """결측치와 이상치 보간을 번갈아 진행하는 파이프라인"""

    def __init__(self, missing_model, outlier_model, device="cuda", iterations=3):
        """
        초기화
        Args:
            missing_model: 결측치 보간 모델
            outlier_model: 이상치 보간 모델
            device: 연산 디바이스
            iterations: 반복 횟수
        """
        super().__init__(missing_model, outlier_model, device)
        self.iterations = iterations
        self.name = f"번갈아 보간 파이프라인 ({iterations}회 반복)"

    def impute(self, data_dict):
        """
        결측치와 이상치 보간을 번갈아 반복적으로 진행
        Args:
            data_dict: 데이터 딕셔너리
        Returns:
            imputed_data: 보간된 데이터
        """
        # 원본 데이터 복사
        data_copy = copy.deepcopy(data_dict)

        # 반복적으로 보간 수행
        for i in range(self.iterations):
            # 결측치 보간
            missing_imputed = self.missing_model.impute(data_copy, device=self.device)
            data_copy["mixed_data"] = missing_imputed

            # 이상치 보간
            outlier_imputed = self.outlier_model.impute(data_copy, device=self.device)
            data_copy["mixed_data"] = outlier_imputed

        return data_copy["mixed_data"]
