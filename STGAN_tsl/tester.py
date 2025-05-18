import logging
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.utils.data as data


class Tester:
    def __init__(self, dataset, model, time_step, node_step, batch_size, n_feature, n_channel, is_shuffle=False):
        """STGAN 모델 테스터 클래스

        Args:
            dataset: STGAN 데이터셋 객체
            model: STGAN 모델 (Generator)
            time_step: 시간 윈도우 크기
            node_step: 노드 스텝 크기
            batch_size: 배치 크기
            n_feature: 특성 수
            n_channel: 채널 수
            is_shuffle: 데이터셋 셔플 여부
        """
        self.dataset = dataset
        self.model = model
        self.time_step = time_step
        self.node_step = node_step
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.n_channel = n_channel

        # 데이터 로더 생성
        self.data_loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=0, drop_last=False
        )

        # 판별자 모델 초기화
        from gan_model import Discriminator

        # 모델 구성 옵션
        self.opt = {
            "num_feature": n_feature * n_channel,
            "hidden_dim": 64,
            "num_layer": 2,
            "num_adj": dataset.n_adj,
            "recent_time": 1,
            "timestamp": time_step,
            "trend_time": 7 * 24,
        }

        # 판별자 모델 인스턴스화
        self.discriminator = Discriminator(self.opt)

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def test(self, checkpoint_path):
        """모델 테스트 함수

        Args:
            checkpoint_path: 체크포인트 경로

        Returns:
            mse: 평균 제곱 오차
            mae: 평균 절대 오차
            rmse: 평균 제곱근 오차
        """
        logging.info(f"테스트 시작: 체크포인트 {checkpoint_path}")

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 적절한 키를 가지고 있는지 확인
        if "generator_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["generator_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        else:
            # 단일 모델 체크포인트로 간주
            self.model.load_state_dict(checkpoint)

        # 평가 모드 설정
        self.model.eval()
        self.discriminator.eval()

        # 예측 결과와 실제 값 저장
        all_preds = []
        all_targets = []
        all_mse_losses = []

        # 테스트 실행
        with torch.no_grad():
            for step, ((recent_data, trend_data, time_feature), subgraph, real_data) in enumerate(self.data_loader):
                # 데이터를 디바이스로 이동
                recent_data = recent_data.to(self.device)
                trend_data = trend_data.to(self.device)
                time_feature = time_feature.to(self.device)
                subgraph = subgraph.to(self.device)
                real_data = real_data.to(self.device)

                # 생성자로 데이터 생성
                fake_data = self.model(recent_data, trend_data, subgraph, time_feature)

                # MSE 계산
                mse_loss = torch.pow(fake_data - real_data, 2).mean(dim=-1)  # 각 노드별 MSE

                # 결과 저장
                all_preds.append(fake_data.cpu().numpy())
                all_targets.append(real_data.cpu().numpy())
                all_mse_losses.append(mse_loss.cpu().numpy())

                # 진행 상태 출력
                if (step + 1) % 10 == 0:
                    logging.info(f"테스트 진행 중: {step+1}/{len(self.data_loader)} 배치 완료")

        # 모든 예측치와 실제값 연결
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_mse_losses = np.concatenate(all_mse_losses, axis=0)

        # 전체 지표 계산
        mse = mean_squared_error(all_targets.reshape(-1), all_preds.reshape(-1))
        mae = mean_absolute_error(all_targets.reshape(-1), all_preds.reshape(-1))
        rmse = np.sqrt(mse)

        logging.info("테스트 완료")
        logging.info(f"MSE: {mse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")

        return mse, mae, rmse

    def detect_outliers(self, checkpoint_path, output_dir, threshold=0.1, convert_to_nan=False):
        """이상치 탐지 함수

        Args:
            checkpoint_path: 체크포인트 경로
            output_dir: 결과 저장 디렉토리
            threshold: 이상치 탐지 임계값
            convert_to_nan: NaN으로 변환할지 여부

        Returns:
            outlier_mask: 이상치 마스크
            mse_values: MSE 값
        """
        logging.info(f"이상치 탐지 시작: 임계값 {threshold}")

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 적절한 키를 가지고 있는지 확인
        if "generator_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["generator_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        else:
            # 단일 모델 체크포인트로 간주
            self.model.load_state_dict(checkpoint)

        # 평가 모드 설정
        self.model.eval()
        self.discriminator.eval()

        # 결과 저장을 위한 텐서 초기화
        n_samples = len(self.dataset)
        n_nodes = self.dataset.n_nodes
        time_steps = n_samples // n_nodes

        mse_values = np.zeros((time_steps, n_nodes))
        outlier_mask = np.zeros((time_steps, n_nodes), dtype=np.int32)

        # 테스트 실행
        with torch.no_grad():
            for step, ((recent_data, trend_data, time_feature), subgraph, real_data) in enumerate(self.data_loader):
                # 데이터를 디바이스로 이동
                recent_data = recent_data.to(self.device)
                trend_data = trend_data.to(self.device)
                time_feature = time_feature.to(self.device)
                subgraph = subgraph.to(self.device)
                real_data = real_data.to(self.device)

                # 생성자로 데이터 생성
                fake_data = self.model(recent_data, trend_data, subgraph, time_feature)

                # MSE 계산
                mse_loss = torch.pow(fake_data - real_data, 2).mean(dim=-1)  # 각 노드별 MSE

                # 배치 내 각 샘플의 인덱스 계산
                batch_size = recent_data.shape[0]
                for b in range(batch_size):
                    idx = step * self.batch_size + b
                    if idx >= n_samples:
                        continue

                    # 시간 및 노드 인덱스 계산
                    time_idx = idx // n_nodes
                    node_idx = idx % n_nodes

                    # MSE 값 저장
                    mse_values[time_idx, node_idx] = torch.mean(mse_loss[b]).item()

                # 진행 상태 출력
                if (step + 1) % 10 == 0:
                    logging.info(f"이상치 탐지 진행 중: {step+1}/{len(self.data_loader)} 배치 완료")

        # 이상치 마스크 생성 (임계값보다 큰 MSE를 가진 데이터 포인트)
        outlier_mask = (mse_values > threshold).astype(np.int32)

        # 이상치 통계
        n_outliers = np.sum(outlier_mask)
        outlier_ratio = n_outliers / outlier_mask.size

        logging.info("이상치 탐지 결과:")
        logging.info(f"전체 데이터 포인트: {outlier_mask.size}")
        logging.info(f"이상치 개수: {n_outliers}")
        logging.info(f"이상치 비율: {outlier_ratio:.4f}")

        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 이상치 마스크 저장
        np.save(os.path.join(output_dir, "outlier_mask.npy"), outlier_mask)

        # MSE 값 저장
        np.save(os.path.join(output_dir, "outlier_mse.npy"), mse_values)

        # 이상치를 NaN으로 변환한 데이터 생성 및 저장
        if convert_to_nan:
            # 원본 데이터 로드
            original_data = self.dataset.data.numpy()

            # 복사본 생성
            nan_data = original_data.copy()

            # 이상치 위치에 NaN 설정
            for t in range(time_steps):
                for n in range(n_nodes):
                    if outlier_mask[t, n] == 1:
                        nan_data[t + self.time_step, n, :, :] = np.nan  # window_size 오프셋 추가

            # NaN 변환된 데이터 저장
            np.save(os.path.join(output_dir, "data_with_nan.npy"), nan_data)
            logging.info("이상치가 NaN으로 변환된 데이터가 저장되었습니다.")

        # 메타데이터 저장
        with open(os.path.join(output_dir, "outlier_metadata.txt"), "w") as f:
            f.write(f"전체 데이터 포인트: {outlier_mask.size}\n")
            f.write(f"이상치 개수: {n_outliers}\n")
            f.write(f"이상치 비율: {outlier_ratio:.4f}\n")
            f.write(f"이상치 탐지 임계값: {threshold}\n")
            f.write(f"이상치를 NaN으로 변환: {convert_to_nan}\n")

        logging.info(f"이상치 탐지 완료. 결과가 {output_dir}에 저장되었습니다.")

        return outlier_mask, mse_values
