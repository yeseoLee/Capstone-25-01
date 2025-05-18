import logging
import os

from load_data import data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


class Tester(object):
    def __init__(self, opt):
        self.opt = opt

        self.loader = data_loader(opt)
        self.generator = data.DataLoader(self.loader, batch_size=opt["batch_size"], shuffle=True, drop_last=False)

        # model
        self.G = torch.load(self.opt["save_path"] + "G_" + str(self.opt["epoch"]) + ".pth")
        self.D = torch.load(self.opt["save_path"] + "D_" + str(self.opt["epoch"]) + ".pth")

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

        # 이상치 탐지 임계값
        self.outlier_threshold = opt.get("outlier_threshold", 0.1)

        # 이상치를 NaN으로 변환할지 여부
        self.convert_outliers = opt.get("convert_outliers", False)

    def test(self):
        self.G.eval()
        self.D.eval()
        result = torch.zeros((self.loader.time_num, self.loader.node_num, 3))
        for step, ((recent_data, trend_data, time_feature), sub_graph, real_data, index_t, index_r) in enumerate(
            self.generator
        ):
            """
            recent_data: (batch_size, time, node_num, num_feature)
            trend_data: (batch_size, time, num_feature)
            real_data: (batch_size, num_adj, num_feature)
            """
            if self.opt["cuda"]:
                recent_data, trend_data, real_data, sub_graph, time_feature = (
                    recent_data.cuda(),
                    trend_data.cuda(),
                    real_data.cuda(),
                    sub_graph.cuda(),
                    time_feature.cuda(),
                )

            # (batch_size, time, num_adj, input_size)
            real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)
            fake_data = self.G(recent_data, trend_data, sub_graph, time_feature)

            fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)
            mse_loss = torch.pow(fake_data - real_data, 2)

            real_score_D = self.D(real_sequence, sub_graph, trend_data)
            fake_score_D = self.D(fake_sequence, sub_graph, trend_data)

            batch_size = recent_data.shape[0]
            for b in range(batch_size):
                result[index_t[b].item(), index_r[b].item(), 0] = torch.mean(mse_loss[b,]).item()
                result[index_t[b].item(), index_r[b].item(), 1] = real_score_D[b].item()
                result[index_t[b].item(), index_r[b].item(), 2] = fake_score_D[b].item()

            if step % 100 == 0:
                logging.info("step:%d [G mse: %f]" % (step, torch.mean(mse_loss)))

        # 기존 결과 저장
        np.save(self.opt["result_path"] + "result" + ".npy", result.cpu().numpy())

        # 이상치 탐지 및 NaN 변환 처리
        self._detect_and_save_outliers(result.cpu().numpy())

    def _detect_and_save_outliers(self, result):
        """이상치를 탐지하고 NaN으로 변환하여 저장합니다.

        Args:
            result: STGAN 모델의 결과 (time, node, 3) - [0]: MSE, [1]: real_score_D, [2]: fake_score_D
        """
        # 원본 데이터 로드
        data_path = self.opt["data_path"] + "/data.npy"
        original_data = np.load(data_path)

        # MSE 값을 기준으로 이상치 탐지
        mse_values = result[:, :, 0]

        # 이상치 마스크 생성 (임계값보다 큰 MSE를 가진 데이터 포인트)
        outlier_mask = (mse_values > self.outlier_threshold).astype(np.int32)

        # 이상치 통계
        n_outliers = np.sum(outlier_mask)
        n_total = outlier_mask.size
        outlier_ratio = n_outliers / n_total

        logging.info("이상치 탐지 결과:")
        logging.info(f"전체 데이터 포인트: {n_total}")
        logging.info(f"이상치 개수: {n_outliers}")
        logging.info(f"이상치 비율: {outlier_ratio:.4f}")

        # 결과 저장 경로
        output_dir = self.opt["result_path"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 이상치 마스크 저장
        np.save(output_dir + "outlier_mask.npy", outlier_mask)

        # MSE 값 저장
        np.save(output_dir + "outlier_mse.npy", mse_values)

        # 이상치를 NaN으로 변환한 데이터 생성 및 저장
        if self.convert_outliers:
            # 복사본 생성
            nan_data = original_data.copy()

            # 이상치 위치 변환
            time_steps, n_nodes = outlier_mask.shape
            # 원본 데이터가 4D일 경우 (시간, 노드, 특성, 채널)
            _, _, n_features, n_channels = original_data.shape

            # 이상치 위치에 NaN 설정
            for t in range(time_steps):
                for n in range(n_nodes):
                    if outlier_mask[t, n] == 1:
                        nan_data[t, n, :, :] = np.nan

            # NaN 변환된 데이터 저장
            np.save(output_dir + "data_with_nan.npy", nan_data)
            logging.info(f"이상치가 NaN으로 변환된 데이터가 {output_dir + 'data_with_nan.npy'}에 저장되었습니다.")

        # 메타데이터 저장
        with open(output_dir + "outlier_metadata.txt", "w") as f:
            f.write(f"전체 데이터 포인트: {n_total}\n")
            f.write(f"이상치 개수: {n_outliers}\n")
            f.write(f"이상치 비율: {outlier_ratio:.4f}\n")
            f.write(f"이상치 탐지 임계값: {self.outlier_threshold}\n")
            f.write(f"이상치를 NaN으로 변환: {self.convert_outliers}\n")

        logging.info(f"이상치 관련 파일들이 {output_dir}에 저장되었습니다.")
