import logging
import os

import torch
import torch.nn as nn
import torch.utils.data as data


class Trainer:
    def __init__(self, dataset, model, time_step, node_step, batch_size, n_feature, n_channel, is_shuffle=True):
        """STGAN 모델 트레이너 클래스

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
        self.data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)

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
            "lambda_G": 500,  # Generator 손실 가중치
        }

        # 판별자 모델 인스턴스화
        self.discriminator = Discriminator(self.opt)

        # 손실 함수 정의
        self.g_loss_fn = nn.MSELoss()
        self.d_loss_fn = nn.BCELoss()

        # 각 모델의 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.g_loss_fn = self.g_loss_fn.to(self.device)
        self.d_loss_fn = self.d_loss_fn.to(self.device)

        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def train(self, model_dir, epochs=100, lr=0.001, verbose_iter=10):
        """모델 학습 함수

        Args:
            model_dir: 모델 저장 디렉토리
            epochs: 학습 에폭 수
            lr: 학습률
            verbose_iter: 로그 출력 간격
        """
        # 옵티마이저 설정
        g_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # 모델 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)

        # 학습 상태 로깅
        logging.info(f"학습 시작: 에폭 {epochs}, 배치 크기 {self.batch_size}, 디바이스 {self.device}")

        # 학습 모드 설정
        self.model.train()
        self.discriminator.train()

        # 에폭별 학습
        for epoch in range(1, epochs + 1):
            g_losses, d_losses = [], []

            # 배치별 학습
            for step, ((recent_data, trend_data, time_feature), subgraph, real_data) in enumerate(self.data_loader):
                # 데이터를 디바이스로 이동
                recent_data = recent_data.to(self.device)
                trend_data = trend_data.to(self.device)
                time_feature = time_feature.to(self.device)
                subgraph = subgraph.to(self.device)
                real_data = real_data.to(self.device)

                # 실제/가짜 레이블 생성
                batch_size = recent_data.shape[0]
                real_label = torch.zeros((batch_size, 1), device=self.device)
                fake_label = torch.ones((batch_size, 1), device=self.device)

                # -----------------------
                # 판별자 학습
                # -----------------------
                d_optimizer.zero_grad()

                # 실제 시퀀스 생성
                real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)

                # 생성자로 가짜 데이터 생성
                fake_data = self.model(recent_data, trend_data, subgraph, time_feature)

                # 가짜 시퀀스 생성
                fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

                # 판별자 예측
                real_pred = self.discriminator(real_sequence, subgraph, trend_data)
                fake_pred = self.discriminator(fake_sequence.detach(), subgraph, trend_data)

                # 판별자 손실 계산
                d_real_loss = self.d_loss_fn(real_pred, real_label)
                d_fake_loss = self.d_loss_fn(fake_pred, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2

                # 역전파 및 최적화
                d_loss.backward()
                d_optimizer.step()

                # -----------------------
                # 생성자 학습
                # -----------------------
                g_optimizer.zero_grad()

                # 가짜 데이터 재생성
                fake_data = self.model(recent_data, trend_data, subgraph, time_feature)
                fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

                # 판별자로 예측
                fake_pred = self.discriminator(fake_sequence, subgraph, trend_data)

                # 생성자 손실 계산 (MSE + 이진 분류)
                g_mse_loss = self.g_loss_fn(fake_data, real_data)
                g_bin_loss = self.d_loss_fn(fake_pred, real_label)
                g_loss = self.opt["lambda_G"] * g_mse_loss + g_bin_loss

                # 역전파 및 최적화
                g_loss.backward()
                g_optimizer.step()

                # 손실 저장
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                # 학습 상태 출력
                if (step + 1) % verbose_iter == 0:
                    # 판별자 정확도 계산
                    d_acc = 0
                    for r_pred in real_pred:
                        if r_pred.item() < 0.5:  # 실제를 실제로 분류
                            d_acc += 1
                    for f_pred in fake_pred:
                        if f_pred.item() > 0.5:  # 가짜를 가짜로 분류
                            d_acc += 1

                    d_acc = d_acc / (batch_size * 2) * 100

                    logging.info(
                        f"에폭 {epoch}/{epochs}, 스텝 {step+1}/{len(self.data_loader)} - "
                        f"D 손실: {d_loss.item():.4f}, D 정확도: {d_acc:.2f}%, "
                        f"G MSE: {g_mse_loss.item():.4f}, G 이진: {g_bin_loss.item():.4f}"
                    )

            # 에폭 평균 손실
            epoch_g_loss = sum(g_losses) / len(g_losses)
            epoch_d_loss = sum(d_losses) / len(d_losses)

            logging.info(f"에폭 {epoch}/{epochs} 완료 - " f"G 손실: {epoch_g_loss:.4f}, D 손실: {epoch_d_loss:.4f}")

            # 모델 저장
            checkpoint_path = os.path.join(model_dir, f"stgan_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": self.model.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "g_loss": epoch_g_loss,
                    "d_loss": epoch_d_loss,
                },
                checkpoint_path,
            )

            # 마지막 모델 저장
            if epoch == epochs:
                final_model_path = os.path.join(model_dir, "stgan_final.pt")
                torch.save(
                    {
                        "generator_state_dict": self.model.state_dict(),
                        "discriminator_state_dict": self.discriminator.state_dict(),
                    },
                    final_model_path,
                )
                logging.info(f"최종 모델 저장 완료: {final_model_path}")

        logging.info(f"학습 완료. 모델이 {model_dir}에 저장되었습니다.")
