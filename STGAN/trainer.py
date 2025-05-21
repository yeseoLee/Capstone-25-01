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

        self.generator = data.DataLoader(data_loader(opt), batch_size=opt["batch_size"], shuffle=True)

        # model
        self.G = Generator(opt)
        self.D = Discriminator(opt)

        # loss function
        self.G_loss = nn.MSELoss()
        self.D_loss = nn.BCELoss()

        if opt["cuda"]:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.G_loss = self.G_loss.cuda()
            self.D_loss = self.D_loss.cuda()

        # Optimizer
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt["lr"])
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt["lr"])

    def train(self):
        self.G.train()
        self.D.train()

        for e in range(1, self.opt["epoch"] + 1):
            for step, ((recent_data, trend_data, time_feature), sub_graph, real_data, _, _) in enumerate(self.generator):
                """
                recent_data: (batch_size, time, node_num, num_feature)
                trend_data: (batch_size, time, num_feature)
                real_data: (batch_size, num_adj, num_feature)
                """

                # 모든 입력 데이터를 CUDA 디바이스로 이동
                if self.opt["cuda"]:
                    recent_data = recent_data.cuda()
                    trend_data = trend_data.cuda()
                    real_data = real_data.cuda()
                    sub_graph = sub_graph.cuda()
                    time_feature = time_feature.cuda()

                # 레이블을 적절한 디바이스로 이동
                valid = torch.ones((real_data.shape[0], 1), dtype=torch.float)
                fake = torch.zeros((real_data.shape[0], 1), dtype=torch.float)
                if self.opt["cuda"]:
                    valid = valid.cuda()
                    fake = fake.cuda()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.D_optim.zero_grad()
                real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)  # (batch_size, time, num_adj, input_size)
                fake_data = self.G(recent_data, trend_data, sub_graph, time_feature)

                fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

                real_score_D = self.D(real_sequence, sub_graph, trend_data)
                fake_score_D = self.D(fake_sequence, sub_graph, trend_data)

                # 입력값을 0과 1 사이로 조정
                real_score_D = torch.clamp(real_score_D, 0, 1)
                fake_score_D = torch.clamp(fake_score_D, 0, 1)

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
                fake_score = torch.clamp(fake_score, 0, 1)  # 입력값을 0과 1 사이로 조정

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
                    print(f"epoch:{e} step:{step} [D loss: {D_total.cpu():f} D acc: {acc*100:.2f}] " f"[G mse: {mse_loss:f} G binary {binary_loss:f}]")

            # 저장 디렉토리 확인
            save_path = self.opt["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 모델 저장
            torch.save(self.G.state_dict(), os.path.join(save_path, f"G_{e}.pth"))
            torch.save(self.D.state_dict(), os.path.join(save_path, f"D_{e}.pth"))
