import json
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

        # 로거 설정
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("STGAN-Tester")

        # 데이터 로드
        self.loader = data_loader(opt)
        self.generator = data.DataLoader(self.loader, batch_size=opt["batch_size"], shuffle=True, drop_last=False)

        # 노드 서브셋 정보 저장
        self.selected_nodes = self.loader.get_selected_nodes()

        # 노드 서브셋 정보 로깅
        if self.selected_nodes is not None:
            self.logger.info(f"노드 서브셋 사용: {len(self.selected_nodes)}개 노드")
        else:
            self.logger.info("전체 노드 사용")

        # 모델 파일 경로
        g_model_path = os.path.join(self.opt["save_path"], f"G_{self.opt['epoch']}.pth")
        d_model_path = os.path.join(self.opt["save_path"], f"D_{self.opt['epoch']}.pth")

        # 모델 파일 존재 확인
        if not os.path.exists(g_model_path):
            self.logger.error(f"생성자 모델 파일이 존재하지 않습니다: {g_model_path}")
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {g_model_path}")
        if not os.path.exists(d_model_path):
            self.logger.error(f"판별자 모델 파일이 존재하지 않습니다: {d_model_path}")
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {d_model_path}")

        # 저장된 모델 로드 시도
        try:
            # 새로운 형식으로 저장된 모델 로드 시도 (dictionary)
            g_state = torch.load(g_model_path)
            d_state = torch.load(d_model_path)

            # 모델 인스턴스 생성 및 상태 로드
            from gan_model import Discriminator, Generator

            self.G = Generator(opt)
            self.D = Discriminator(opt)

            # 모델 상태 로드
            if isinstance(g_state, dict) and "model_state_dict" in g_state:
                # 새 형식 (노드 서브셋 정보 포함)
                self.G.load_state_dict(g_state["model_state_dict"])
                self.D.load_state_dict(d_state["model_state_dict"])

                # 노드 서브셋 정보 확인
                if "node_subset_info" in g_state:
                    self.logger.info("모델에서 노드 서브셋 정보를 로드했습니다.")
                    if "selected_nodes" in g_state["node_subset_info"]:
                        loaded_nodes = g_state["node_subset_info"]["selected_nodes"]
                        if self.selected_nodes is None:
                            self.logger.warning("모델에는 노드 서브셋 정보가 있지만, 현재 설정에서는 전체 노드를 사용합니다.")
                        elif set(self.selected_nodes) != set(loaded_nodes):
                            self.logger.warning("현재 설정의 노드 서브셋이 모델의 노드 서브셋과 다릅니다. 이는 테스트 결과에 영향을 줄 수 있습니다.")
            else:
                # 기존 형식 (모델 자체)
                self.G = g_state
                self.D = d_state
                self.logger.info("기존 형식의 모델을 로드했습니다.")
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            # 기존 방식으로 로드 시도
            self.G = torch.load(g_model_path)
            self.D = torch.load(d_model_path)
            self.logger.info("모델을 기존 방식으로 로드했습니다.")

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
        self.convert_outliers = opt.get("convert-outliers", False)

    def test(self):
        self.G.eval()
        self.D.eval()
        result = torch.zeros((self.loader.time_num, self.loader.node_num, 3))

        self.logger.info(f"테스트 시작 - 시간 단계: {self.loader.time_num}, 노드 수: {self.loader.node_num}")

        for step, ((recent_data, trend_data, time_feature), sub_graph, real_data, index_t, index_r) in enumerate(self.generator):
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
                self.logger.info(f"step:{step} [G mse: {torch.mean(mse_loss):f}]")

        # 결과 저장 경로 생성
        if not os.path.exists(self.opt["result_path"]):
            os.makedirs(self.opt["result_path"])

        # 노드 서브셋 정보가 포함된 파일명 생성
        if self.selected_nodes is not None:
            result_filename = f"result_nodes_{len(self.selected_nodes)}"
        else:
            result_filename = "result"

        # 기존 결과 저장
        result_path = os.path.join(self.opt["result_path"], f"{result_filename}.npy")
        np.save(result_path, result.cpu().numpy())
        self.logger.info(f"테스트 결과가 {result_path}에 저장되었습니다.")

        # 노드 서브셋 정보가 포함된 메타데이터 저장
        metadata = {
            "test_info": {"time_steps": self.loader.time_num, "node_count": self.loader.node_num, "original_node_count": self.loader.original_node_num},
            "outlier_detection": {"threshold": self.outlier_threshold, "convert_to_nan": self.convert_outliers},
        }

        # 노드 서브셋 정보 추가
        if self.selected_nodes is not None:
            metadata["node_subset_info"] = {"use_node_subset": True, "selected_nodes_count": len(self.selected_nodes)}

            # 노드 수가 100개 이하면 선택된 노드 목록 저장
            if len(self.selected_nodes) <= 100:
                metadata["node_subset_info"]["selected_nodes"] = self.selected_nodes

            # 노드 서브셋 정보 저장
            node_info_path = os.path.join(self.opt["result_path"], f"nodes_info_{len(self.selected_nodes)}.json")
            with open(node_info_path, "w") as f:
                node_info = {"selected_nodes_count": len(self.selected_nodes), "selected_nodes": self.selected_nodes}
                json.dump(node_info, f, indent=2)
        else:
            metadata["node_subset_info"] = {"use_node_subset": False, "comment": "전체 노드 사용"}

        # 메타데이터 저장
        metadata_path = os.path.join(self.opt["result_path"], f"{result_filename}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 이상치 탐지 및 NaN 변환 처리
        self._detect_and_save_outliers(result.cpu().numpy(), result_filename)

    def _detect_and_save_outliers(self, result, result_filename="result"):
        """이상치를 탐지하고 NaN으로 변환하여 저장합니다.

        Args:
            result: STGAN 모델의 결과 (time, node, 3) - [0]: MSE, [1]: real_score_D, [2]: fake_score_D
            result_filename: 결과 파일 이름 접두사
        """
        # 원본 데이터 로드
        data_path = os.path.join(self.opt["data_path"], "data.npy")
        if not os.path.exists(data_path):
            self.logger.error(f"원본 데이터 파일이 존재하지 않습니다: {data_path}")
            self.logger.info("이상치 탐지를 건너뜁니다.")
            return

        original_data = np.load(data_path)

        # MSE 값을 기준으로 이상치 탐지
        mse_values = result[:, :, 0]

        # 이상치 마스크 생성 (임계값보다 큰 MSE를 가진 데이터 포인트)
        outlier_mask = (mse_values > self.outlier_threshold).astype(np.int32)

        # 이상치 통계
        n_outliers = np.sum(outlier_mask)
        n_total = outlier_mask.size
        outlier_ratio = n_outliers / n_total

        self.logger.info("이상치 탐지 결과:")
        self.logger.info(f"전체 데이터 포인트: {n_total}")
        self.logger.info(f"이상치 개수: {n_outliers}")
        self.logger.info(f"이상치 비율: {outlier_ratio:.4f}")

        # 결과 저장 경로
        output_dir = self.opt["result_path"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 이상치 마스크 저장
        outlier_mask_path = os.path.join(output_dir, f"{result_filename}_outlier_mask.npy")
        np.save(outlier_mask_path, outlier_mask)

        # MSE 값 저장
        mse_path = os.path.join(output_dir, f"{result_filename}_outlier_mse.npy")
        np.save(mse_path, mse_values)

        # 이상치를 NaN으로 변환한 데이터 생성 및 저장
        if self.convert_outliers:
            # 복사본 생성
            nan_data = original_data.copy()

            # 이상치 위치 변환
            time_steps, n_nodes = outlier_mask.shape
            # 원본 데이터가 4D일 경우 (시간, 노드, 특성, 채널)
            _, _, n_features, n_channels = original_data.shape

            # 노드 서브셋을 사용하는 경우 원본 데이터의 해당 노드에만 적용
            if self.selected_nodes is not None:
                for t in range(time_steps):
                    for i, n in enumerate(range(n_nodes)):
                        if outlier_mask[t, n] == 1:
                            original_node_idx = self.selected_nodes[i]
                            nan_data[t, original_node_idx, :, :] = np.nan
            else:
                # 전체 노드를 사용하는 경우
                for t in range(time_steps):
                    for n in range(n_nodes):
                        if outlier_mask[t, n] == 1:
                            nan_data[t, n, :, :] = np.nan

            # NaN 변환된 데이터 저장
            nan_data_path = os.path.join(output_dir, f"{result_filename}_data_with_nan.npy")
            np.save(nan_data_path, nan_data)
            self.logger.info(f"이상치가 NaN으로 변환된 데이터가 {nan_data_path}에 저장되었습니다.")

        # 메타데이터 저장
        outlier_metadata_path = os.path.join(output_dir, f"{result_filename}_outlier_metadata.txt")
        with open(outlier_metadata_path, "w") as f:
            f.write(f"전체 데이터 포인트: {n_total}\n")
            f.write(f"이상치 개수: {n_outliers}\n")
            f.write(f"이상치 비율: {outlier_ratio:.4f}\n")
            f.write(f"이상치 탐지 임계값: {self.outlier_threshold}\n")
            f.write(f"이상치를 NaN으로 변환: {self.convert_outliers}\n")

            # 노드 서브셋 정보 추가
            if self.selected_nodes is not None:
                f.write("\n# 노드 서브셋 정보\n")
                f.write(f"사용 노드 수: {len(self.selected_nodes)}개\n")
                f.write(f"전체 노드 수: {self.loader.original_node_num}개\n")
                f.write(f"사용 비율: {len(self.selected_nodes)/self.loader.original_node_num*100:.1f}%\n")

                # 노드 목록이 너무 길지 않은 경우 출력
                if len(self.selected_nodes) <= 30:
                    f.write(f"선택된 노드: {', '.join(map(str, self.selected_nodes))}\n")
                else:
                    f.write(f"선택된 노드: (총 {len(self.selected_nodes)}개, 별도 파일 참조)\n")

        self.logger.info(f"이상치 관련 파일들이 {output_dir}에 저장되었습니다.")
