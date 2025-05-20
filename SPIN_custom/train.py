"""
SPIN 모델을 학습하는 스크립트
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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


def get_model(model_str="spin", **kwargs):
    """모델을 초기화합니다."""
    if model_str == "spin":
        model = SPINModel
    else:
        raise ValueError(f"Model {model_str} not available.")

    return model(**kwargs)


def train(args):
    """SPIN 모델을 학습합니다."""
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

    # 모델 초기화
    model = get_model(model_str=args.model, input_size=dataset.n_channels, hidden_size=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout)

    # 콜백 설정
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
        ModelCheckpoint(dirpath=args.save_dir, filename="{epoch:02d}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1),
    ]

    # 로거 설정
    logger = TensorBoardLogger(args.log_dir, name=args.model)

    # 트레이너 초기화
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=callbacks, logger=logger, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    # 학습 실행
    trainer.fit(model, data_module)

    # 최종 모델 저장
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"최종 모델이 저장되었습니다: {final_model_path}")


if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument("--dataset", type=str, default="bay", help="데이터셋 이름")
    parser.add_argument("--model", type=str, default="spin", help="모델 이름")
    parser.add_argument("--hidden_size", type=int, default=64, help="은닉층 크기")
    parser.add_argument("--n_layers", type=int, default=2, help="레이어 수")
    parser.add_argument("--dropout", type=float, default=0.1, help="드롭아웃 비율")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=100, help="에폭 수")
    parser.add_argument("--patience", type=int, default=10, help="조기 종료 인내심")
    parser.add_argument("--workers", type=int, default=4, help="데이터 로더 워커 수")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="모델 저장 디렉토리")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="로그 디렉토리")
    parser.add_argument("--use-node-subset", action="store_true", help="노드 서브셋 사용")
    parser.add_argument("--node-list", type=str, help="사용할 노드 인덱스 목록 (쉼표로 구분)")

    args = parser.parse_args()
    train(args)
