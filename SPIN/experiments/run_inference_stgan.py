import copy
import os
import sys


# 상위 디렉토리를 경로에 추가해 utils에 접근할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytorch_lightning as pl
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
import torch
import tsl
from tsl import config
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.imputers import Imputer
from tsl.nn.models.imputation import GRINModel
from tsl.nn.utils import casting
from tsl.ops.imputation import add_missing_values, sample_mask
from tsl.utils import ArgParser, numpy_metrics, parser_utils
from tsl.utils.python_utils import ensure_list

# STGAN 데이터셋 로더 가져오기
from utils.stgan_dataset import STGANBayDataset
import yaml


def get_model_classes(model_str):
    if model_str == "spin":
        model, filler = SPINModel, SPINImputer
    elif model_str == "spin_h":
        model, filler = SPINHierarchicalModel, SPINImputer
    elif model_str == "grin":
        model, filler = GRINModel, Imputer
    else:
        raise ValueError(f"Model {model_str} not available.")
    return model, filler


def get_dataset(dataset_name: str, root_dir=None):
    # STGAN 데이터셋 사용
    if dataset_name.endswith("_point"):
        p_fault, p_noise = 0.0, 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith("_block"):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")

    if dataset_name == "bay":
        # STGAN 데이터 형식의 Bay 데이터셋 로드
        stgan_dataset = STGANBayDataset(root_dir=root_dir).load()
        # 결측치 추가
        return add_missing_values(
            stgan_dataset, p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=56789
        )

    raise ValueError(f"Invalid dataset name: {dataset_name}.")


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--config", type=str, default="inference.yaml")
    parser.add_argument("--root", type=str, default="log")
    parser.add_argument("--root-dir", type=str, default=None, help="STGAN 데이터셋 경로")
    parser.add_argument("--output-dir", type=str, default="imputed_data", help="보정된 데이터를 저장할 경로")

    # Data sparsity params
    parser.add_argument("--p-fault", type=float, default=0.0)
    parser.add_argument("--p-noise", type=float, default=0.75)
    parser.add_argument("--test-mask-seed", type=int, default=None)

    # Splitting/aggregation params
    parser.add_argument("--val-len", type=float, default=0.1)
    parser.add_argument("--test-len", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, "r") as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def load_model(exp_dir, exp_config, dm):
    model_cls, imputer_class = get_model_classes(exp_config["model_name"])
    additional_model_hparams = {
        "n_nodes": dm.n_nodes,
        "input_size": dm.n_channels,
        "u_size": 31,  # STGAN time_features는 31개 (7일 + 24시간)
        "output_size": dm.n_channels,
        "window_size": dm.window,
    }

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**exp_config, **additional_model_hparams}, target_cls=model_cls, return_dict=True
    )

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(exp_config, imputer_class, return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={},
        loss_fn=None,
        **imputer_kwargs,
    )

    model_path = None
    for file in os.listdir(exp_dir):
        if file.endswith(".ckpt"):
            model_path = os.path.join(exp_dir, file)
            break
    if model_path is None:
        raise ValueError("Model not found.")

    imputer.load_model(model_path)
    imputer.freeze()
    return imputer


def update_test_eval_mask(dm, dataset, p_fault, p_noise, seed=None):
    if seed is None:
        seed = np.random.randint(1e9)
    random = np.random.default_rng(seed)
    dataset.set_eval_mask(sample_mask(dataset.shape, p=p_fault, p_noise=p_noise, min_seq=12, max_seq=36, rng=random))
    dm.torch_dataset.set_mask(dataset.training_mask)
    dm.torch_dataset.update_exogenous("eval_mask", dataset.eval_mask)


def save_imputed_data_stgan_format(args, imputed_data, original_data, mask, dataset):
    """결측치가 보정된 데이터를 STGAN 형식으로 저장합니다.

    Args:
        args: 명령줄 인수
        imputed_data: 결측치가 보정된 데이터 (시간, 노드, 채널) 3차원
        original_data: 원본 데이터 (시간, 노드, 채널) 3차원
        mask: 마스크 데이터 (결측치 위치, 1 = 관측값, 0 = 결측값)
        dataset: STGAN 데이터셋 객체
    """
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # STGAN 원본 데이터 형태 가져오기
    original_stgan_data = dataset.data
    time_steps, n_nodes, n_features, n_channels = original_stgan_data.shape

    # 3D 텐서를 4D 텐서로 변환 (STGAN 형식에 맞게)
    # TSL: (시간, 노드, 특성*채널) -> STGAN: (시간, 노드, 특성, 채널)

    # 마스킹된 위치에 보정된 값으로 대체
    # 마스크가 0인 위치(결측값)에 보정된 값 사용, 1인 위치(관측값)에 원본 값 사용
    combined_data = np.where(mask.reshape(time_steps, n_nodes, 1) == 1, original_data, imputed_data)

    # 3D -> 4D로 reshape
    imputed_stgan_shaped = combined_data.reshape(time_steps, n_nodes, n_features, n_channels)

    # 데이터 저장
    output_path = os.path.join(output_dir, "imputed_data.npy")
    np.save(output_path, imputed_stgan_shaped)

    # 원본 데이터와 보정된 데이터의 차이 계산 및 저장
    masked_diff = np.where(mask.reshape(time_steps, n_nodes, 1) == 0, np.abs(imputed_data - original_data), 0)
    masked_diff_reshaped = masked_diff.reshape(time_steps, n_nodes, n_features, n_channels)
    np.save(os.path.join(output_dir, "imputation_diff.npy"), masked_diff_reshaped)

    # 마스크 저장
    np.save(os.path.join(output_dir, "mask.npy"), mask.reshape(time_steps, n_nodes, 1))

    # 결측 통계 저장
    n_missing = np.sum(mask == 0)
    n_total = mask.size
    missing_ratio = n_missing / n_total

    # 메타데이터 저장
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        f.write(f"Total data points: {n_total}\n")
        f.write(f"Missing data points: {n_missing}\n")
        f.write(f"Missing ratio: {missing_ratio:.4f}\n")
        f.write(f"Original shape: {original_stgan_data.shape}\n")
        f.write(f"Imputed shape: {imputed_stgan_shaped.shape}\n")
        f.write(f"Experiment name: {args.exp_name}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset_name}\n")

    # 원본 STGAN 데이터셋 구조에 필요한 파일들 복사
    # 시간 특성, 인접 노드, 거리 정보 등은 변경되지 않으므로 원본 파일 사용
    import shutil

    # time_features.txt 복사
    src_time_features = os.path.join(dataset.data_dir, "time_features.txt")
    dst_time_features = os.path.join(output_dir, "time_features.txt")
    if os.path.exists(src_time_features):
        shutil.copy2(src_time_features, dst_time_features)

    # node_adjacent.txt 복사
    src_node_adjacent = os.path.join(dataset.data_dir, "node_adjacent.txt")
    dst_node_adjacent = os.path.join(output_dir, "node_adjacent.txt")
    if os.path.exists(src_node_adjacent):
        shutil.copy2(src_node_adjacent, dst_node_adjacent)

    # node_dist.txt 복사
    src_node_dist = os.path.join(dataset.data_dir, "node_dist.txt")
    dst_node_dist = os.path.join(output_dir, "node_dist.txt")
    if os.path.exists(src_node_dist):
        shutil.copy2(src_node_dist, dst_node_dist)

    # node_subgraph.npy 복사
    src_node_subgraph = os.path.join(dataset.data_dir, "node_subgraph.npy")
    dst_node_subgraph = os.path.join(output_dir, "node_subgraph.npy")
    if os.path.exists(src_node_subgraph):
        shutil.copy2(src_node_subgraph, dst_node_subgraph)

    print(f"보정된 데이터가 {output_dir} 디렉토리에 저장되었습니다.")
    print("파일 목록:")
    print(" - imputed_data.npy: 보정된 데이터 (원본 형식)")
    print(" - imputation_diff.npy: 원본과 보정 데이터의 차이 (결측치 위치에서만)")
    print(" - mask.npy: 결측치 마스크")
    print(" - metadata.txt: 메타데이터")
    print(" - time_features.txt: 시간 특성 (복사됨)")
    print(" - node_adjacent.txt: 인접 노드 정보 (복사됨)")
    print(" - node_dist.txt: 노드 간 거리 정보 (복사됨)")
    if os.path.exists(dst_node_subgraph):
        print(" - node_subgraph.npy: 노드 서브그래프 정보 (복사됨)")

    return output_path


def run_experiment(args):
    # Set configuration
    args = copy.deepcopy(args)
    tsl.logger.disabled = True

    # script flags
    is_spin = args.model_name in ["spin", "spin_h"]

    ########################################
    # load config                          #
    ########################################

    if args.root is None:
        root = tsl.config.log_dir
    else:
        root = os.path.join(tsl.config.curr_dir, args.root)
    exp_dir = os.path.join(root, args.dataset_name, args.model_name, args.exp_name)

    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        exp_config = yaml.load(fp, Loader=yaml.FullLoader)

    ########################################
    # load dataset                         #
    ########################################

    dataset = get_dataset(exp_config["dataset_name"], root_dir=args.root_dir)

    ########################################
    # load data module                     #
    ########################################

    # time embedding
    if is_spin:
        time_emb = dataset.datetime_encoded(["day", "week"]).values()
        exog_map = {"global_temporal_encoding": time_emb}

        input_map = {"u": "temporal_encoding", "x": "data"}
    else:
        exog_map = input_map = None

    if is_spin or args.model_name == "grin":
        adj = dataset.get_connectivity(threshold=args.adj_threshold, include_self=False, force_symmetric=is_spin)
    else:
        adj = None

    # instantiate dataset
    torch_dataset = ImputationDataset(
        *dataset.numpy(return_idx=True),
        training_mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        connectivity=adj,
        exogenous=exog_map,
        input_map=input_map,
        window=exp_config["window"],
        stride=exp_config["stride"],
    )

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    scalers = {"data": StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset, scalers=scalers, splitter=splitter, batch_size=args.batch_size)
    dm.setup()

    ########################################
    # load model                           #
    ########################################

    imputer = load_model(exp_dir, exp_config, dm)

    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()))

    ########################################
    # inference                            #
    ########################################

    # 테스트 마스크 설정
    seeds = ensure_list(args.test_mask_seed)
    if not seeds:
        seeds = [None]  # 시드가 지정되지 않은 경우 기본값 사용

    # 각 시드에 대해 예측 실행
    for seed in seeds:
        # 데이터 로드 및 마스크 설정
        update_test_eval_mask(dm, dataset, args.p_fault, args.p_noise, seed)

        # 예측 실행
        output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
        output = casting.numpy(output)

        # 결과 추출
        y_hat = output["y_hat"].squeeze(-1)  # 보정값
        y_true = output["y"].squeeze(-1)  # 원본값
        mask = output["mask"].squeeze(-1)  # 마스크

        # MAE 계산
        check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
        print(f"SEED {seed} - Test MAE: {check_mae:.2f}")

        # 원본 스케일로 역변환 (표준화 해제)
        # 스케일러가 사용된 경우 원래 스케일로 다시 변환
        if hasattr(dm, "scalers") and "data" in dm.scalers:
            scaler = dm.scalers["data"]
            y_hat = scaler.inverse_transform(y_hat)
            y_true = scaler.inverse_transform(y_true)

        # 결과 저장 (STGAN 형식)
        if seed is not None:
            output_path = os.path.join(args.output_dir, f"seed_{seed}")
        else:
            output_path = args.output_dir

        # 결과를 STGAN 형식으로 저장
        args.output_dir = output_path
        save_imputed_data_stgan_format(args, y_hat, y_true, mask, dataset)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
