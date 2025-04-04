from abc import ABC, abstractmethod
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class BaseModel(nn.Module, ABC):
    """보간 모델의 기본 클래스"""

    def __init__(self, name=None):
        super(BaseModel, self).__init__()
        self.name = name if name is not None else self.__class__.__name__

    @abstractmethod
    def forward(self, data):
        """순전파"""
        pass

    def train_model(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=100,
        device="cuda",
        early_stop_patience=10,
        model_save_path=None,
        verbose=True,
    ):
        """
        모델 학습
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            optimizer: 옵티마이저
            criterion: 손실 함수
            epochs: 에폭 수
            device: 학습 디바이스
            early_stop_patience: Early stopping 인내 횟수
            model_save_path: 모델 저장 경로
            verbose: 로그 출력 여부

        Returns:
            best_model: 가장 좋은 성능의 모델
            history: 학습 히스토리
        """
        self.to(device)
        self.train()

        # 학습 히스토리
        history = {"train_loss": [], "val_loss": []}

        # Early stopping 설정
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # 학습
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not verbose):
                # 데이터를 디바이스로 이동
                batch = batch.to(device)

                # 그래디언트 초기화
                optimizer.zero_grad()

                # 순전파
                outputs = self(batch)

                # 손실 계산 (결측치/이상치 마스크 위치만)
                combined_mask = batch.missing_mask | batch.outlier_mask
                loss = criterion(outputs[combined_mask], batch.y[combined_mask])

                # 역전파 및 최적화
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 에폭 평균 손실
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # 검증
            val_loss = self.evaluate(val_loader, criterion, device)
            history["val_loss"].append(val_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping 확인
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.state_dict().copy()

                # 최고 성능 모델 저장
                if model_save_path is not None:
                    torch.save(best_model_state, model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        # 최고 성능 모델 로드
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self, history

    def evaluate(self, data_loader, criterion=None, device="cuda"):
        """
        모델 평가
        Args:
            data_loader: 데이터 로더
            criterion: 손실 함수 (None이면 MSE 사용)
            device: 평가 디바이스

        Returns:
            avg_loss: 평균 손실
        """
        if criterion is None:
            criterion = nn.MSELoss()

        self.to(device)
        self.eval()

        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                # 데이터를 디바이스로 이동
                batch = batch.to(device)

                # 순전파
                outputs = self(batch)

                # 손실 계산 (결측치/이상치 마스크 위치만)
                combined_mask = batch.missing_mask | batch.outlier_mask
                loss = criterion(outputs[combined_mask], batch.y[combined_mask])

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def impute(self, data_dict, device="cuda"):
        """
        데이터 보간
        Args:
            data_dict: 데이터 딕셔너리
            device: 평가 디바이스

        Returns:
            imputed_data: 보간된 데이터
        """
        self.to(device)
        self.eval()

        # 데이터 추출
        mixed_data = data_dict["mixed_data"]
        missing_mask = data_dict["missing_mask"]
        outlier_mask = data_dict["outlier_mask"]
        adj_mx = data_dict["adj_mx"]

        # PyG 데이터 객체 생성
        edge_index = torch.tensor(np.array(np.nonzero(adj_mx)), dtype=torch.long).to(device)

        # 보간할 데이터 복사
        imputed_data = mixed_data.copy()

        # PyTorch 텐서로 변환
        x = torch.tensor(mixed_data, dtype=torch.float32).to(device)

        # 추론
        with torch.no_grad():
            # 모델 입력 준비
            batch = type("obj", (), {})()
            batch.x = x
            batch.edge_index = edge_index
            batch.missing_mask = torch.tensor(missing_mask, dtype=torch.bool).to(device)
            batch.outlier_mask = torch.tensor(outlier_mask, dtype=torch.bool).to(device)

            # 추론
            outputs = self(batch)

            # 결과를 NumPy 배열로 변환
            outputs_np = outputs.cpu().numpy()

            # 결측치 및 이상치 위치에 보간 값 적용
            combined_mask = missing_mask | outlier_mask
            imputed_data[combined_mask] = outputs_np[combined_mask]

        return imputed_data

    def save(self, path):
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_name": self.name,
            },
            path,
        )

    def load(self, path, device="cuda"):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.name = checkpoint.get("model_name", self.name)
        return self
