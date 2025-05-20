"""
SPIN 모델을 테스트하는 스크립트
"""

import os

import numpy as np
import pytorch_lightning as pl
from spin.models import SPINModel
import torch
from tsl import logger
from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.utils.parser_utils import ArgParser


def get_dataset(dataset_name, root_dir=None, selected_nodes=None):
    """데이터셋을 로드합니다."""
    from experiments.run_imputation import DirectSTGANDataset

    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), "datasets", dataset_name)

    dataset = DirectSTGANDataset(root_dir=root_dir, selected_nodes=selected_nodes)
    dataset.load()

    return dataset


def test(args):
    """SPIN 모델을 테스트합니다."""
    # 노드 리스트 파싱
    selected_nodes = None
    if args.node_list:
        selected_nodes = [int(x) for x in args.node_list.split(",")]
        logger.info(f"선택된 노드: {selected_nodes}")

    # 데이터셋 로드
    dataset = get_dataset(args.dataset, selected_nodes=selected_nodes)

    # 데이터 모듈 초기화
    data_module = SpatioTemporalDataModule(
        dataset=dataset, scalers={"target": StandardScaler(axis=(0, 1))}, splitter=dataset.get_splitter(method="temporal"), batch_size=args.batch_size, workers=args.workers
    )

    # 모델 로드
    model = SPINModel.load_from_checkpoint(args.model_path, input_size=dataset.n_channels, hidden_size=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout)

    # 트레이너 초기화
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    # 테스트 실행
    results = trainer.test(model, data_module)

    # 결과 저장
    output_dir = os.path.join(args.root_path, f"datasets/{args.dataset}/spin_results")
    os.makedirs(output_dir, exist_ok=True)

    # 예측 결과 저장
    predictions = trainer.predict(model, data_module)
    predictions = torch.cat(predictions, dim=0).numpy()

    output_path = os.path.join(output_dir, "imputed_data.npy")
    np.save(output_path, predictions)
    logger.info(f"보정된 데이터가 저장되었습니다: {output_path}")

    # 평가 지표 출력
    for metric_name, value in results[0].items():
        logger.info(f"{metric_name}: {value:.4f}")

    return output_path, results[0]["test_mae"], results[0]["test_mse"]


if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument("--dataset", type=str, default="bay", help="데이터셋 이름")
    parser.add_argument("--model_path", type=str, required=True, help="학습된 모델 경로")
    parser.add_argument("--hidden_size", type=int, default=64, help="은닉층 크기")
    parser.add_argument("--n_layers", type=int, default=2, help="레이어 수")
    parser.add_argument("--dropout", type=float, default=0.1, help="드롭아웃 비율")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--workers", type=int, default=4, help="데이터 로더 워커 수")
    parser.add_argument("--root_path", type=str, default="./", help="루트 경로")
    parser.add_argument("--use-node-subset", action="store_true", help="노드 서브셋 사용")
    parser.add_argument("--node-list", type=str, help="사용할 노드 인덱스 목록 (쉼표로 구분)")

    args = parser.parse_args()
    test(args)
