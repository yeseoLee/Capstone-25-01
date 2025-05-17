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
from spin.imputers import SPINImputer
from spin.models import SPINHierarchicalModel, SPINModel
from spin.stgan_dataset import STGANBayDataset
import torch
from torch.utils.data import DataLoader
import tsl
from tsl import config
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.utils import ArgParser, numpy_metrics, parser_utils
from tsl.utils.python_utils import ensure_list
import yaml


def get_model_classes(model_str):
    if model_str == "spin":
        model, filler = SPINModel, SPINImputer
    elif model_str == "spin_h":
        model, filler = SPINHierarchicalModel, SPINImputer
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


# TSL Data 객체를 안전하게 처리하는 커스텀 콜레이터 함수
def safe_collate_fn(batch):
    """TSL Data 객체를 안전하게 처리하는 콜레이터 함수

    Args:
        batch: 배치 데이터

    Returns:
        배치 데이터 (TSL Data 객체가 그대로 반환됨)
    """
    # 배치 크기가 1인 경우 그대로 반환
    if len(batch) == 1:
        return batch[0]

    # 여러 TSL Data 객체를 처리 (원래 TSL에서 정의한 방식으로)
    # 이 부분은 데이터셋 형식에 따라 다르게 구현해야 할 수 있음
    print(f"배치 크기: {len(batch)}, 형식: {type(batch[0])}")
    return batch[0]  # 일단 첫 번째 항목만 반환


def run_experiment(args):  # noqa: C901
    # Set configuration
    args = copy.deepcopy(args)
    tsl.logger.disabled = True

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
    time_emb = dataset.datetime_encoded(["day", "week"]).values()
    exog_map = {"global_temporal_encoding": time_emb}
    input_map = {"u": "temporal_encoding", "x": "data"}

    adj = dataset.get_connectivity(threshold=args.adj_threshold, include_self=False, force_symmetric=True)
    # PyTorch Geometric을 위한 edge_index 생성 (long 타입)
    rows, cols = np.where(adj > 0)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

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
        # 시드가 없으면 기본값 하나만 사용 (메모리 문제 해결을 위해)
        seeds = [1234]

    # Colab이나 제한된 메모리 환경에서 실행할 때 더 작은 배치 크기 사용
    test_batch_size = min(args.batch_size, 4)  # 배치 크기 제한 (더 작게 설정)

    print(f"테스트 배치 크기: {test_batch_size}")
    print(f"사용할 시드 목록: {seeds}")

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

        # TSL 라이브러리의 기본 데이터로더 사용 (커스텀 데이터로더 대신)
        # 배치 크기 조정을 위해 DataModule 설정 업데이트
        if hasattr(dm, "batch_size") and dm.batch_size > test_batch_size:
            print(f"배치 크기 조정: {dm.batch_size} -> {test_batch_size}")
            dm.batch_size = test_batch_size
            # 데이터로더 재설정
            dm.setup()

        # 원본 TSL 데이터로더 가져오기
        original_dataloader = dm.test_dataloader()
        print(f"원본 테스트 데이터셋 크기: {len(original_dataloader.dataset)}")

        # 배치 크기가 1인 새로운 데이터로더 생성 (메모리 문제 방지)
        # 배치 크기를 1로 설정하면 collate_fn이 필요 없음
        test_dataloader = DataLoader(
            original_dataloader.dataset,
            batch_size=1,  # 배치 크기를 1로 설정하여 collate 문제 방지
            shuffle=False,
            num_workers=0,  # 메모리 문제 방지를 위해 worker 수 제한
            pin_memory=False,  # 메모리 사용량 최적화
            collate_fn=safe_collate_fn,  # 안전한 콜레이트 함수 사용
        )

        print(f"테스트 데이터로더 생성 완료: 배치 크기=1, 배치 수={len(test_dataloader)}")

        try:
            # 예측 전 메모리 사용량 확인
            print(get_gpu_memory_usage())

            # PyTorch Lightning의 predict 메서드 대신 직접 구현
            print("PyTorch Lightning Trainer 대신 직접 예측 수행...")

            # 모델을 평가 모드로 설정
            imputer.model.eval()

            # 결과 저장용 리스트
            all_y_hat = []
            all_y = []
            all_mask = []

            # 배치별 예측
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    # 첫 10개 배치마다 진행 상황 출력
                    if batch_idx % 10 == 0:
                        print(f"배치 {batch_idx}/{len(test_dataloader)} 처리 중... {get_gpu_memory_usage()}")

                    # 배치 데이터를 GPU로 이동 (가능한 경우)
                    if torch.cuda.is_available():
                        batch = batch.to("cuda")

                    # 예측 수행
                    # TSL 데이터 객체인 경우 내부 데이터 접근
                    if hasattr(batch, "x") and hasattr(batch, "y") and hasattr(batch, "mask"):
                        try:
                            # 모델 예측
                            y_hat = imputer.predict_batch(batch)

                            # 결과 저장
                            all_y_hat.append(y_hat.detach().cpu())
                            all_y.append(batch.y.detach().cpu())
                            all_mask.append(batch.mask.detach().cpu())
                        except Exception as e:
                            print(f"배치 {batch_idx} 예측 중 오류 발생: {e}")
                            # 배치 구조 디버깅
                            print(f"배치 구조: {[attr for attr in dir(batch) if not attr.startswith('_')]}")
                    else:
                        print(f"배치 {batch_idx}에 필요한 속성이 없음. 건너뜀.")

                    # 메모리 관리
                    if batch_idx % 20 == 0:
                        torch.cuda.empty_cache()

            # 결과 통합
            if all_y_hat:
                # 텐서 연결
                y_hat = torch.cat(all_y_hat, dim=0)
                y = torch.cat(all_y, dim=0)
                mask = torch.cat(all_mask, dim=0)

                # NumPy 배열로 변환
                y_hat_np = y_hat.numpy()
                y_np = y.numpy()
                mask_np = mask.numpy()

                print(f"예측 완료! 총 {len(all_y_hat)} 배치 처리됨.")
                print(f"결과 형태: y_hat={y_hat_np.shape}, y={y_np.shape}, mask={mask_np.shape}")

                # MAE 계산
                check_mae = numpy_metrics.masked_mae(y_hat_np, y_np, mask_np)
                print(f"SEED {seed} - Test MAE: {check_mae:.2f}")

                # 원본 스케일로 역변환 (표준화 해제)
                # 스케일러가 사용된 경우 원래 스케일로 다시 변환
                if hasattr(dm, "scalers") and "data" in dm.scalers:
                    scaler = dm.scalers["data"]
                    y_hat_np = scaler.inverse_transform(y_hat_np)
                    y_np = scaler.inverse_transform(y_np)

                # 결과 저장 (STGAN 형식)
                if seed is not None:
                    output_path = os.path.join(args.output_dir, f"seed_{seed}")
                else:
                    output_path = args.output_dir

                # 결과를 STGAN 형식으로 저장
                args.output_dir = output_path
                save_imputed_data_stgan_format(args, y_hat_np, y_np, mask_np, dataset)
            else:
                print("경고: 예측 결과가 없습니다.")
                continue
        except Exception as e:
            print(f"오류 발생: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
