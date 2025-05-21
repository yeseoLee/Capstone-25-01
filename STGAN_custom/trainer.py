import json
import logging
import os

from gan_model import Discriminator, Generator
from load_data import data_loader
import torch
import torch.nn as nn
import torch.utils.data as data


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 데이터 로더 초기화
        self.loader = data_loader(opt)
        self.generator = data.DataLoader(self.loader, batch_size=opt["batch_size"], shuffle=True)

        # 노드 서브셋 정보 저장
        self.selected_nodes = self.loader.get_selected_nodes()

        # 로거 설정
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("STGAN-Trainer")

        # 노드 서브셋 정보 로깅
        if self.selected_nodes is not None:
            self.logger.info(f"노드 서브셋 사용: {len(self.selected_nodes)}개 노드")
        else:
            self.logger.info("전체 노드 사용")

        # model
        self.G = Generator(opt)
        self.D = Discriminator(opt)

        # loss function
        self.G_loss = nn.MSELoss()
        self.D_loss = nn.BCELoss()

        # 모델과 손실 함수를 적절한 디바이스로 이동
        if opt["cuda"]:
            self.logger.info(f"모델을 {self.device}로 이동합니다.")
            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)
            self.G_loss = self.G_loss.to(self.device)
            self.D_loss = self.D_loss.to(self.device)
        else:
            self.logger.info("CPU 모드로 실행합니다.")

        # Optimizer
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt["lr"])
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt["lr"])

    def D_loss(self, score, label):
        """
        Discriminator의 손실 함수
        Args:
            score: 판별자의 출력값
            label: 실제 레이블 (1: 진짜, 0: 가짜)
        """
        # 입력값을 0과 1 사이로 조정
        score = torch.clamp(score, 0, 1)
        label = torch.clamp(label, 0, 1)

        criterion = nn.BCELoss()
        return criterion(score, label)

    def train(self):
        self.G.train()
        self.D.train()

        # 노드 서브셋 정보를 모델 체크포인트에 저장하기 위한 정보 준비
        node_info = {}
        if self.selected_nodes is not None:
            node_info = {
                "use_node_subset": True,
                "selected_nodes_count": len(self.selected_nodes),
            }
            # 노드 수가 너무 많으면 체크포인트 파일이 너무 커지지 않도록 제한
            if len(self.selected_nodes) <= 100:
                node_info["selected_nodes"] = self.selected_nodes
            else:
                node_info["selected_nodes_info"] = f"총 {len(self.selected_nodes)}개 노드 (별도 파일 참조)"

        for e in range(1, self.opt["epoch"] + 1):
            for step, ((recent_data, trend_data, time_feature), sub_graph, real_data, _, _) in enumerate(self.generator):
                """
                recent_data: (batch_size, time, node_num, num_feature)
                trend_data: (batch_size, time, num_feature)
                real_data: (batch_size, num_adj, num_feature)
                """

                valid = torch.zeros((real_data.shape[0], 1), dtype=torch.float)
                fake = torch.ones((real_data.shape[0], 1), dtype=torch.float)

                # 데이터를 적절한 디바이스로 이동
                if self.opt["cuda"]:
                    recent_data = recent_data.to(self.device)
                    trend_data = trend_data.to(self.device)
                    real_data = real_data.to(self.device)
                    sub_graph = sub_graph.to(self.device)
                    time_feature = time_feature.to(self.device)
                    valid = valid.to(self.device)
                    fake = fake.to(self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.D_optim.zero_grad()
                real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)  # (batch_size, time, num_adj, input_size)
                fake_data = self.G(recent_data, trend_data, sub_graph, time_feature)

                fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

                real_score_D = self.D(real_sequence, sub_graph, trend_data)
                fake_score_D = self.D(fake_sequence, sub_graph, trend_data)

                real_loss = self.D_loss(real_score_D, valid)
                fake_loss = self.D_loss(fake_score_D, fake)
                D_total = (real_loss + fake_loss) / 2

                D_total.backward(retain_graph=True)
                self.D_optim.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.G_optim.zero_grad()
                fake_data = self.G(recent_data, trend_data, sub_graph, time_feature)

                mse_loss = self.G_loss(fake_data, real_data)
                fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

                fake_score = self.D(fake_sequence, sub_graph, trend_data)

                binary_loss = self.D_loss(fake_score, valid)
                G_total = self.opt["lambda_G"] * mse_loss + binary_loss

                G_total.backward()
                self.G_optim.step()

                if step % 100 == 0:
                    count = 0
                    for score in real_score_D:
                        if torch.mean(score) < 0.5:
                            count += 1
                    for score in fake_score_D:
                        if torch.mean(score) > 0.5:
                            count += 1

                    acc = count / (self.opt["batch_size"] * 2)
                    self.logger.info(f"epoch:{e} step:{step} [D loss: {D_total.cpu():f} D acc: {acc*100:.2f}] " f"[G mse: {mse_loss:f} G binary {binary_loss:f}]")

            # 저장 디렉토리 확인
            save_path = self.opt["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 노드 서브셋 정보를 메타데이터 파일로 저장
            metadata_file = os.path.join(save_path, f"epoch_{e}_metadata.json")
            metadata = {
                "epoch": e,
                "model_info": {"hidden_dim": self.opt["hidden_dim"], "num_adj": self.opt["num_adj"], "num_layer": self.opt["num_layer"]},
                "training_params": {"lr": self.opt["lr"], "batch_size": self.opt["batch_size"], "lambda_G": self.opt["lambda_G"]},
                "node_subset_info": node_info,
                "device_info": {"device": str(self.device), "use_cuda": self.opt["cuda"]},
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # 모델 저장
            # 노드 서브셋 정보를 추가하여 저장
            g_state = {"model_state_dict": self.G.state_dict(), "node_subset_info": node_info}
            d_state = {"model_state_dict": self.D.state_dict(), "node_subset_info": node_info}

            # 모델 파일 경로
            g_path = os.path.join(self.opt["save_path"], f"G_{e}.pth")
            d_path = os.path.join(self.opt["save_path"], f"D_{e}.pth")
            torch.save(g_state, g_path)
            torch.save(d_state, d_path)

            self.logger.info(f"Epoch {e} 모델 및 메타데이터 저장 완료: {g_path}, {d_path}")
