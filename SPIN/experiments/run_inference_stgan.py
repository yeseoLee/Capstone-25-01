"""
STGAN 데이터셋에 대한 SPIN 모델 추론 스크립트

이 스크립트는 훈련된 SPIN 모델을 사용하여 STGAN 데이터셋에 대한 결측치 보간을 수행합니다.
메모리 최적화 및 진행 상황 모니터링을 위한 기능이 포함되어 있습니다.

사용법:
    python experiments/run_inference_stgan.py --model-name spin_h --dataset-name bay_point
    --exp-name [실험명] --root-dir [STGAN 데이터셋 경로] --output-dir [출력 경로]

추가 옵션:
    --p-fault: 블록 결측 확률 (기본값: 0.0)
    --p-noise: 포인트 결측 확률 (기본값: 0.75)
    --batch-size: 배치 크기 (기본값: 32, 메모리 문제 시 더 작은 값 사용)
"""

import copy
import os
import traceback

import numpy as np
import pytorch_lightning as pl
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
from spin.stgan_dataset import STGANBayDataset
import torch
from torch.utils.data import DataLoader
import tsl
from tsl import config
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.imputers import Imputer
from tsl.nn.models.imputation import GRINModel
from tsl.nn.utils import casting
from tsl.utils import ArgParser, numpy_metrics, parser_utils
from tsl.utils.python_utils import ensure_list
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

        # 결측치를 직접 추가하는 로직 구현
        # 원본 데이터 복사
        data = stgan_dataset._target.copy()
        mask = np.ones_like(data, dtype=bool)

        # 시드 설정
        seed = 56789
        rng = np.random.RandomState(seed)

        # 데이터 크기
        time_steps, n_nodes, n_channels = data.shape

        # 포인트 결측치 추가 (p_noise 확률로 랜덤하게 데이터 포인트 마스킹)
        if p_noise > 0:
            noise_mask = rng.rand(time_steps, n_nodes, n_channels) < p_noise
            mask = mask & ~noise_mask

        # 블록 결측치 추가 (p_fault 확률로 각 노드의 연속된 블록 마스킹)
        if p_fault > 0:
            # 각 노드에 대해 독립적으로 처리
            for n in range(n_nodes):
                # p_fault 확률로 결측이 시작되는 시점들 결정
                fault_points = rng.rand(time_steps) < p_fault
                fault_indices = np.where(fault_points)[0]

                # 각 결측 시작점에 대해 min_seq에서 max_seq 사이의 랜덤한 길이로 마스킹
                min_seq, max_seq = 12, 12 * 4
                for idx in fault_indices:
                    if idx >= time_steps:
                        continue

                    # 랜덤 길이 결정
                    seq_len = rng.randint(min_seq, max_seq + 1)
                    end_idx = min(idx + seq_len, time_steps)

                    # 모든 채널에 대해 마스킹
                    mask[idx:end_idx, n, :] = False

        # 결측치 적용 (마스킹된 부분을 NaN으로 설정)
        masked_data = data.copy()
        masked_data[~mask] = np.nan

        # 데이터셋 업데이트
        stgan_dataset._target = masked_data
        stgan_dataset._mask = mask
        stgan_dataset._training_mask = mask.copy()
        stgan_dataset._eval_mask = mask.copy()

        return stgan_dataset

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
        # 추가로 확실하게 설정
        "h_size": dm.n_channels,  # 채널 수와 일치하도록 h_size 설정
        "z_size": dm.n_channels,  # 채널 수와 일치하도록 z_size 설정
    }

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**exp_config, **additional_model_hparams}, target_cls=model_cls, return_dict=True
    )

    # 모델 경로 찾기
    model_path = None
    for file in os.listdir(exp_dir):
        if file.endswith(".ckpt"):
            model_path = os.path.join(exp_dir, file)
            break
    if model_path is None:
        raise ValueError("Model not found.")

    # 모델 경로가 있으면 직접 로드
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # 모델 생성
    model = model_cls(**model_kwargs)

    # 모델 파라미터 로드
    state_dict = checkpoint["state_dict"]

    # 'model.'로 시작하는 키를 찾아서 접두사 제거
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v  # 'model.' 접두사 제거
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("가중치를 직접 로드할 수 없습니다. 대신 새 모델을 초기화합니다.")

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

    # 모델 설정
    imputer.model = model
    imputer.freeze()

    return imputer


def update_test_eval_mask(dm, dataset, p_fault, p_noise, seed=None):
    """테스트용 평가 마스크를 업데이트합니다.

    Args:
        dm: 데이터 모듈
        dataset: 데이터셋
        p_fault: 블록 결측 확률
        p_noise: 포인트 결측 확률
        seed: 랜덤 시드
    """
    if seed is None:
        seed = np.random.randint(1e9)

    # 데이터 크기
    time_steps, n_nodes, n_channels = dataset.shape

    # 마스크 초기화 (전체 True로 시작)
    mask = np.ones((time_steps, n_nodes, n_channels), dtype=bool)

    # 랜덤 생성기 초기화
    rng = np.random.RandomState(seed)

    # 포인트 결측치 추가 (p_noise 확률로 랜덤하게 데이터 포인트 마스킹)
    if p_noise > 0:
        noise_mask = rng.rand(time_steps, n_nodes, n_channels) < p_noise
        mask = mask & ~noise_mask

    # 블록 결측치 추가 (p_fault 확률로 각 노드의 연속된 블록 마스킹)
    if p_fault > 0:
        # 각 노드에 대해 독립적으로 처리
        for n in range(n_nodes):
            # p_fault 확률로 결측이 시작되는 시점들 결정
            fault_points = rng.rand(time_steps) < p_fault
            fault_indices = np.where(fault_points)[0]

            # 각 결측 시작점에 대해 min_seq에서 max_seq 사이의 랜덤한 길이로 마스킹
            min_seq, max_seq = 12, 36  # 원래 샘플 함수의 값 사용
            for idx in fault_indices:
                if idx >= time_steps:
                    continue

                # 랜덤 길이 결정
                seq_len = rng.randint(min_seq, max_seq + 1)
                end_idx = min(idx + seq_len, time_steps)

                # 모든 채널에 대해 마스킹
                mask[idx:end_idx, n, :] = False

    # 평가 마스크 업데이트
    dataset.set_eval_mask(mask)

    # 데이터 모듈 업데이트
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


def run_experiment(args):  # noqa: C901
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
        # PyTorch Geometric을 위한 edge_index 생성 (long 타입)
        rows, cols = np.where(adj > 0)
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    else:
        adj = None
        edge_index = None

    # instantiate dataset
    torch_dataset = ImputationDataset(
        *dataset.numpy(return_idx=True),
        training_mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        connectivity=adj,  # 여기서는 adj 행렬 사용
        exogenous=exog_map,
        input_map=input_map,
        window=exp_config["window"],
        stride=exp_config["stride"],
    )

    # PyTorch Geometric을 위한 edge_index를 torch_dataset에 저장
    if edge_index is not None:
        torch_dataset.edge_index = edge_index

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    # TSL 라이브러리의 SpatioTemporalDataModule 클래스는 splitter.split() 메서드를 기대합니다.
    # 반환된 splitter가 함수인 경우, 이를 처리하기 위한 래퍼 클래스 생성
    class SplitterWrapper:
        def __init__(self, split_fn):
            self.split_fn = split_fn
            self._splits = None

        def split(self, dataset):
            # 데이터셋의 인덱스를 가져오기
            idx = dataset.idx if hasattr(dataset, "idx") else np.arange(len(dataset))
            # 분할 함수 호출
            split_masks = self.split_fn(idx)
            # 각 데이터셋 유형에 해당하는 인덱스 저장
            self._splits = {k: np.where(v)[0] for k, v in split_masks.items()}
            return self._splits

        def get_split(self, split):
            if self._splits is None:
                raise RuntimeError("You must call split() before get_split()")
            return self._splits.get(split, None)

        @property
        def train_idxs(self):
            if self._splits is None:
                raise RuntimeError("You must call split() before accessing train_idxs")
            return self._splits.get("train", None)

        @property
        def val_idxs(self):
            if self._splits is None:
                raise RuntimeError("You must call split() before accessing val_idxs")
            return self._splits.get("val", None)

        @property
        def test_idxs(self):
            if self._splits is None:
                raise RuntimeError("You must call split() before accessing test_idxs")
            return self._splits.get("test", None)

    # splitter가 함수인 경우 래퍼로 감싸기
    if callable(splitter) and not hasattr(splitter, "split"):
        splitter = SplitterWrapper(splitter)

    scalers = {"data": StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset, scalers=scalers, splitter=splitter, batch_size=args.batch_size)
    dm.setup()

    ########################################
    # load model                           #
    ########################################

    imputer = load_model(exp_dir, exp_config, dm)

    # 테스트 마스크 설정
    seeds = ensure_list(args.test_mask_seed)
    if not seeds:
        seeds = [None]  # 시드가 지정되지 않은 경우 기본값 사용

    # Colab이나 제한된 메모리 환경에서 실행할 때 더 작은 배치 크기 사용
    test_batch_size = min(args.batch_size, 8)  # 배치 크기 제한

    # 메모리 최적화를 위한 trainer 설정
    trainer = pl.Trainer(
        gpus=int(torch.cuda.is_available()),
        enable_progress_bar=True,  # 진행 상황 표시
        enable_model_summary=False,  # 모델 요약 비활성화
        enable_checkpointing=False,  # 체크포인트 비활성화
    )

    print(f"테스트 배치 크기: {test_batch_size}")

    # GPU 메모리 사용량 확인 함수
    def get_gpu_memory_usage():
        if not torch.cuda.is_available():
            return "GPU not available"

        # 현재 장치
        device = torch.cuda.current_device()

        # 할당된 메모리
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB 단위

        # 캐시된 메모리
        cached_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB 단위

        # 전체 메모리
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB 단위

        return (
            f"GPU Memory: {allocated_memory:.2f}GB allocated, {cached_memory:.2f}GB cached, {total_memory:.2f}GB total"
        )

    # 각 시드에 대해 예측 실행
    for seed in seeds:
        # 데이터 로드 및 마스크 설정
        update_test_eval_mask(dm, dataset, args.p_fault, args.p_noise, seed)

        # 테스트 데이터로더 생성 (배치 크기 조정)
        test_dataset = dm.test_dataloader().dataset
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,  # 메모리 문제 방지를 위해 worker 수 제한
            pin_memory=False,  # 메모리 사용량 최적화
        )

        try:
            # 예측 전 메모리 사용량 확인
            print(get_gpu_memory_usage())

            # 예측 실행 (PyTorch Lightning이 내부적으로 배치 처리)
            print(f"시드 {seed}에 대한 예측 시작...")
            # 메모리 사용량 모니터링 메시지 추가
            print("예측 중... (멈춘 것처럼 보일 수 있으나 배치 처리 중입니다)")

            # 사용자 정의 콜백 함수 생성 (메모리 모니터링용)
            class MemoryMonitorCallback(pl.Callback):
                def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                    # 매 5 배치마다 메모리 사용량 출력
                    if batch_idx % 5 == 0:
                        print(f"Batch {batch_idx} 완료 - {get_gpu_memory_usage()}")

                    # 메모리 부족 문제 방지를 위해 주기적으로 캐시 비우기
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

            # 메모리 모니터링 콜백 추가
            memory_callback = MemoryMonitorCallback()
            trainer.callbacks.append(memory_callback)

            # 예측 실행
            output = trainer.predict(imputer, dataloaders=test_dataloader)
            print(f"예측 완료! {get_gpu_memory_usage()}")

            # 메모리 해제
            torch.cuda.empty_cache()
            print(f"메모리 정리 후: {get_gpu_memory_usage()}")

            # 예측 결과 확인
            if output is None or len(output) == 0:
                print(f"경고: 시드 {seed}에 대한 예측 결과가 비어 있습니다. 다음 시드로 넘어갑니다.")
                continue

            output = casting.numpy(output)

            # 결과 추출
            y_hat = output["y_hat"]  # 보정값
            y_true = output["y"]  # 원본값
            mask = output["mask"]  # 마스크

            # 마지막 차원이 1인 경우에만 squeeze 적용
            if y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            if y_true.shape[-1] == 1:
                y_true = y_true.squeeze(-1)
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)

            # MAE 계산
            check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
            print(f"SEED {seed} - Test MAE: {check_mae:.2f}")

            # 원본 스케일로 역변환 (표준화 해제)
            # 스케일러가 사용된 경우 원래 스케일로 다시 변환
            if hasattr(dm, "scalers") and "data" in dm.scalers:
                scaler = dm.scalers["data"]
                # 텐서를 NumPy 배열로 변환
                if hasattr(y_hat, "numpy"):
                    y_hat = y_hat.numpy()
                if hasattr(y_true, "numpy"):
                    y_true = y_true.numpy()
                # 스케일러 역변환 적용
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
        except Exception as e:
            print(f"오류 발생: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
