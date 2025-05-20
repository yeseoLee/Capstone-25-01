import argparse
import os
import random

import numpy as np
from tester import Tester
import torch
from trainer import Trainer


torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="bay", help="bay")
parser.add_argument("--root_path", type=str, default="./", help="root path: dataset, checkpoint")

parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
parser.add_argument("--epoch", type=int, default=6, help="Number of training epochs per iteration.")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lambda_G", type=int, default=500, help="lambda_G for generator loss function")

parser.add_argument("--num_adj", type=int, default=9, help="number of nodes in sub graph")
parser.add_argument("--num_layer", type=int, default=2, help="number of layers in LSTM and DCRNN")
parser.add_argument("--trend_time", type=int, default=7 * 24, help="the length of trend segment is 7 days")

# CPU/GPU 설정 옵션 개선
parser.add_argument("--cpu", action="store_true", help="CPU 모드 사용 (기본값: GPU 사용)")
parser.add_argument("--cuda_id", type=str, default="0", help="사용할 CUDA 디바이스 ID")
parser.add_argument("--seed", type=int, default=20)

# 이상치 탐지 관련 인자 추가
parser.add_argument("--outlier_threshold", type=float, default=0.1, help="Threshold for outlier detection based on MSE")
parser.add_argument("--convert-outliers", action="store_true", help="이상치를 NaN으로 변환할지 여부")

# 노드 선택 관련 인자 추가
parser.add_argument("--use-node-subset", action="store_true", help="일부 노드만 사용하여 메모리 사용량 감소")
parser.add_argument("--node-ratio", type=float, default=0.2, help="전체 노드 중 사용할 비율 (0.0-1.0)")
parser.add_argument("--node-list", type=str, default=None, help="사용할 노드 인덱스 목록 (쉼표로 구분)")

args = parser.parse_args()

# 랜덤 시드 설정
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# CUDA 사용 가능 여부 확인 및 설정
cuda_available = torch.cuda.is_available()
if args.cpu:
    use_cuda = False
    device = torch.device("cpu")
    print("CPU 모드로 실행합니다.")
elif cuda_available:
    use_cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device(f"cuda:{args.cuda_id}")
    torch.cuda.manual_seed(args.seed)
    print(f"GPU 모드로 실행합니다. (CUDA 디바이스: {args.cuda_id})")
else:
    use_cuda = False
    device = torch.device("cpu")
    print("CUDA를 사용할 수 없어 CPU 모드로 실행합니다.")

# parameter
opt = vars(args)
opt["cuda"] = use_cuda  # cuda 플래그 설정
opt["device"] = device  # 디바이스 정보 추가

# 2017-01-01 - 2017-05-06
if opt["dataset"] == "bay":
    opt["timestamp"] = 12  # 5min: 12 or 30min: 2
    opt["train_time"] = 105  # days for training
    opt["recent_time"] = 1  # bay: 1 hour, nyc: 2hour
    opt["num_feature"] = 6 * 2  # length of input feature
    opt["time_feature"] = 31  # length of time feature

# 경로 설정 부분 수정 - 이중 슬래시 문제 해결
opt["save_path"] = os.path.join(opt["root_path"], f"datasets/{opt['dataset']}/checkpoint/")
opt["data_path"] = os.path.join(opt["root_path"], f"datasets/{opt['dataset']}/data")  # 끝에 슬래시 제거
opt["result_path"] = os.path.join(opt["root_path"], f"datasets/{opt['dataset']}/result/")

# 디렉토리 존재 확인 및 생성
for path in [opt["save_path"], os.path.dirname(opt["data_path"]), opt["result_path"]]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"디렉토리 생성: {path}")

opt["train_time"] = opt["train_time"] * opt["timestamp"] * 24

# 노드 비율 유효성 검사
if "node_ratio" in opt and opt["node_ratio"] is not None:
    if opt["node_ratio"] <= 0 or opt["node_ratio"] > 1.0:
        print(f"경고: 노드 비율({opt['node_ratio']})이 범위를 벗어납니다. 기본값 0.2로 설정합니다.")
        opt["node_ratio"] = 0.2

# 노드 서브셋 처리 - 메모리 및 성능 모니터링을 위한 코드 추가
if opt["use_node_subset"]:
    try:
        import datetime
        import time

        import psutil

        # 시작 시간 및 메모리 사용량 측정
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB 단위

        print(f"실험 시작 - 초기 메모리 사용량: {initial_memory:.2f} MB")

        # 노드 서브셋 정보
        if opt["node_list"]:
            print("노드 서브셋 사용: 사용자 지정 노드 목록")
        else:
            print(f"노드 서브셋 사용: 전체 노드의 {opt['node_ratio']*100:.1f}%")

        # 노드 서브셋 옵션에 따른 선택된 노드 처리
        selected_nodes = None

        # 성능 모니터링 결과 저장 디렉토리 생성
        log_dir = os.path.join(opt["result_path"], "node_subset_logs/")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 결과 파일 이름에 타임스탬프 추가
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        opt["log_file"] = os.path.join(log_dir, f"node_subset_log_{timestamp}.txt")

        with open(opt["log_file"], "w") as f:
            f.write(f"실험 시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"초기 메모리 사용량: {initial_memory:.2f} MB\n")
            f.write(f"디바이스: {device}\n")

            if opt["node_list"]:
                f.write(f"노드 서브셋 설정: 사용자 지정 노드 목록 - {opt['node_list']}\n")
            else:
                f.write(f"노드 서브셋 설정: 전체 노드의 {opt['node_ratio']*100:.1f}%\n")

        # 노드 정보 저장 파일
        opt["node_info_file"] = os.path.join(log_dir, f"selected_nodes_{timestamp}.txt")

    except Exception as e:
        print(f"성능 모니터링 초기화 중 오류 발생: {str(e)}")
        print("성능 모니터링 없이 계속 진행합니다.")
        opt["log_file"] = None
        opt["node_info_file"] = None

if __name__ == "__main__":
    try:
        # 훈련 단계
        opt["isTrain"] = True
        train_model = Trainer(opt)
        train_model.train()

        # 테스트 단계
        opt["isTrain"] = False
        test_model = Tester(opt)
        test_model.test()

        # 노드 서브셋 사용 시 성능 정보 출력
        if opt.get("use_node_subset", False) and "start_time" in locals():
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
            total_time = end_time - start_time

            print("=" * 50)
            print("실험 완료 - 성능 요약")
            print(f"총 실행 시간: {total_time:.2f} 초 ({total_time/60:.2f} 분)")
            print(f"최종 메모리 사용량: {final_memory:.2f} MB")
            print(f"메모리 증가량: {final_memory - initial_memory:.2f} MB")

            # 노드 정보 출력
            if hasattr(train_model, "selected_nodes") and train_model.selected_nodes is not None:
                node_count = len(train_model.selected_nodes)
                total_nodes = train_model.loader.original_node_num
                print(f"노드 서브셋 사용: {node_count}개 노드 ({total_nodes}개 중 {node_count}개 사용, {node_count/total_nodes*100:.1f}%)")
            else:
                print("전체 노드 사용")
            print("=" * 50)

            # 로그 파일에 성능 정보 추가
            if opt.get("log_file") and os.path.exists(opt["log_file"]):
                with open(opt["log_file"], "a") as f:
                    f.write("=" * 50 + "\n")
                    f.write("실험 완료 - 성능 요약\n")
                    f.write(f"실험 종료 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"총 실행 시간: {total_time:.2f} 초 ({total_time/60:.2f} 분)\n")
                    f.write(f"최종 메모리 사용량: {final_memory:.2f} MB\n")
                    f.write(f"메모리 증가량: {final_memory - initial_memory:.2f} MB\n")

                    # 노드 정보 출력
                    if hasattr(train_model, "selected_nodes") and train_model.selected_nodes is not None:
                        node_count = len(train_model.selected_nodes)
                        total_nodes = train_model.loader.original_node_num
                        f.write(f"노드 서브셋 사용: {node_count}개 노드 ({total_nodes}개 중 {node_count}개 사용, {node_count/total_nodes*100:.1f}%)\n")
                    else:
                        f.write("전체 노드 사용\n")
                    f.write("=" * 50 + "\n")

    except Exception as e:
        import traceback

        print(f"실행 중 오류 발생: {str(e)}")
        traceback.print_exc()

        # 오류 정보 로깅
        if opt.get("log_file"):
            with open(opt["log_file"], "a") as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("오류 발생\n")
                f.write(f"오류 메시지: {str(e)}\n")
                f.write(f"오류 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("스택 트레이스:\n")
                import traceback

                traceback.print_exc(file=f)
                f.write("=" * 50 + "\n")
