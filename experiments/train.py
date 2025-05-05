import argparse
import os

import torch

from ..models import (
    GCNMI,  # 결측치 보간 모델
    GCNODE,  # 이상치 보간 모델
)
from ..utils import (
    create_dataloader,
    create_torch_dataset,
    load_dataset,
    split_data,
)


def train_outlier_model(train_data, val_data, args):
    """
    이상치 보간 모델 학습
    Args:
        train_data: 학습 데이터
        val_data: 검증 데이터
        args: 파라미터

    Returns:
        trained_model: 학습된 모델
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 생성
    train_dataset = create_torch_dataset(train_data, window_size=args.window_size)
    val_dataset = create_torch_dataset(val_data, window_size=args.window_size)

    # 데이터로더 생성
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size)

    # 손실 함수
    criterion = torch.nn.MSELoss()

    # 모델 생성 및 학습
    print("\n=== 이상치 보간 모델 학습 ===")

    # GCN 기반 이상치 보간 모델
    print("\nTraining GCNODE...")
    gcnode = GCNODE(
        in_channels=args.window_size,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(gcnode.parameters(), lr=args.learning_rate)
    gcnode, _ = gcnode.train_model(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=args.epochs,
        device=device,
        early_stop_patience=args.patience,
        model_save_path=os.path.join(args.model_dir, "gcnode.pt"),
    )

    return gcnode


def train_missing_model(train_data, val_data, args):
    """
    결측치 보간 모델 학습
    Args:
        train_data: 학습 데이터
        val_data: 검증 데이터
        args: 파라미터

    Returns:
        trained_model: 학습된 모델
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 생성
    train_dataset = create_torch_dataset(train_data, window_size=args.window_size)
    val_dataset = create_torch_dataset(val_data, window_size=args.window_size)

    # 데이터로더 생성
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size)

    # 손실 함수
    criterion = torch.nn.MSELoss()

    # 모델 생성 및 학습
    print("\n=== 결측치 보간 모델 학습 ===")

    # GCN 기반 결측치 보간 모델
    print("\nTraining GCNMI...")
    gcnmi = GCNMI(
        in_channels=args.window_size,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(gcnmi.parameters(), lr=args.learning_rate)
    gcnmi, _ = gcnmi.train_model(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=args.epochs,
        device=device,
        early_stop_patience=args.patience,
        model_save_path=os.path.join(args.model_dir, "gcnmi.pt"),
    )

    return gcnmi


def do_train(args):
    """
    메인 함수
    Args:
        args: 파라미터
    """
    # 디렉토리 생성
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # 데이터셋 로드
    print("데이터셋 로드 중...")
    data_dict = load_dataset(args.data_path)

    # 데이터 분할
    print("데이터 분할 중...")
    train_data, val_data, _ = split_data(data_dict, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # 이상치 보간 모델 학습
    outlier_model = train_outlier_model(train_data, val_data, args)

    # 결측치 보간 모델 학습
    missing_model = train_missing_model(train_data, val_data, args)

    print("\n모델 학습 완료!")
    print(f"모델 저장 경로: {args.model_dir}")

    # 모델 저장
    outlier_model.save(os.path.join(args.model_dir, "gcnode.pt"))
    missing_model.save(os.path.join(args.model_dir, "gcnmi.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="교통 데이터 이상치/결측치 보간 모델 학습")

    # 데이터 관련 인자
    parser.add_argument("--data_path", type=str, default="datasets/processed/traffic_data.pkl", help="데이터셋 경로")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="학습 데이터 비율")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="검증 데이터 비율")

    # 모델 관련 인자
    parser.add_argument("--window_size", type=int, default=12, help="시계열 윈도우 크기")
    parser.add_argument("--hidden_dim", type=int, default=64, help="숨겨진 차원 크기")
    parser.add_argument("--num_layers", type=int, default=2, help="GNN 레이어 수")
    parser.add_argument("--dropout", type=float, default=0.2, help="드롭아웃 비율")
    parser.add_argument("--gat_heads", type=int, default=4, help="GAT 어텐션 헤드 수")

    # 학습 관련 인자
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="학습률")
    parser.add_argument("--patience", type=int, default=10, help="조기 종료 인내심")

    # 저장 관련 인자
    parser.add_argument("--model_dir", type=str, default="models/saved/", help="모델 저장 경로")
    parser.add_argument("--result_dir", type=str, default="results/", help="결과 저장 경로")

    # 인자 파싱
    args = parser.parse_args()

    do_train(args)
