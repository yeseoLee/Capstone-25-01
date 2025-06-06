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

parser.add_argument("--cpu", type=bool, default=False)
parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
parser.add_argument("--cuda_id", type=str, default="3")
parser.add_argument("--seed", type=int, default=20)

# 이상치 탐지 관련 인자 추가
parser.add_argument("--outlier_threshold", type=float, default=0.1, help="Threshold for outlier detection based on MSE")
parser.add_argument("--convert_outliers", type=bool, default=False, help="Whether to convert outliers to NaN")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# parameter
opt = vars(args)
# 2017-01-01 - 2017-05-06
if opt["dataset"] == "bay":
    opt["timestamp"] = 12  # 5min: 12 or 30min: 2
    opt["train_time"] = 105  # days for training
    opt["recent_time"] = 1  # bay: 1 hour, nyc: 2hour
    opt["num_feature"] = 6 * 2  # length of input feature
    opt["time_feature"] = 31  # length of time feature

# 경로 설정 수정 - 프로젝트 최상단의 datasets 폴더 참조
opt["save_path"] = os.path.join(opt["root_path"], f"../datasets/{opt['dataset']}/checkpoint/")
opt["data_path"] = os.path.join(opt["root_path"], f"../datasets/{opt['dataset']}/data/")
opt["result_path"] = os.path.join(opt["root_path"], f"../datasets/{opt['dataset']}/result/")

opt["train_time"] = opt["train_time"] * opt["timestamp"] * 24
if __name__ == "__main__":
    opt["isTrain"] = True
    train_model = Trainer(opt)
    train_model.train()

    opt["isTrain"] = False
    test_model = Tester(opt)
    test_model.test()
