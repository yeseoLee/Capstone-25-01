"""
STGAN 모델을 이용한 PemsBay 데이터셋 실험 스크립트

이 스크립트는 TSL의 PemsBay 데이터셋을 사용하여 STGAN 모델을 학습하고
결측치 보간 성능을 평가합니다.

기본 사용법:
    python run_stgan_pemsbay.py
"""

import argparse
import os

# TSL 데이터셋 변환 모듈 임포트
from convert_format import pemsbay_with_missing_to_stgan_format

# STGAN_tsl 모듈 임포트
from gan_model import Generator
from stgan_dataset import STGANDataset
from tester import Tester
from trainer import Trainer


def prepare_pemsbay_dataset(args):
    """
    PemsBay 데이터셋을 STGAN 형식으로 준비합니다.

    Args:
        args: 명령행 인수

    Returns:
        data_dir: 데이터 디렉토리 경로
    """
    # 결측치 유형에 따른 설정
    if args.missing_type == "block":
        p_fault, p_noise = 0.0015, 0.05
    elif args.missing_type == "point":
        p_fault, p_noise = 0.0, 0.25
    else:
        raise ValueError(f"지원하지 않는 결측치 유형: {args.missing_type}")

    # STGAN 형식으로 변환된 데이터셋 경로
    data_dir = f"./bay/data/stgan_format_{args.missing_type}"

    # 데이터셋이 없으면 생성
    if not os.path.exists(data_dir) or args.force_regenerate:
        print(f"PemsBay 데이터셋을 STGAN 형식으로 변환 중 (유형: {args.missing_type})...")
        pemsbay_with_missing_to_stgan_format(p_fault=p_fault, p_noise=p_noise, seed=args.seed, output_dir=data_dir)
    else:
        print(f"기존 STGAN 형식 데이터셋을 사용합니다: {data_dir}")

    return data_dir


def train_stgan_model(args, data_dir):
    """
    STGAN 모델을 학습합니다.

    Args:
        args: 명령행 인수
        data_dir: 데이터 디렉토리 경로

    Returns:
        model_dir: 모델 저장 디렉토리 경로
    """
    # 데이터셋 로드
    dataset = STGANDataset(data_dir=data_dir)

    # STGAN 모델 생성
    stgan = Generator(
        time_step=args.time_steps,  # 시간 창 크기
        node_count=dataset.n_nodes,  # 노드 수
        feature_size=dataset.n_features,  # 특성 수
        channel_in=dataset.n_channels,  # 입력 채널 수
        node_dim=args.node_dim,  # 노드 임베딩 차원
        dropout=args.dropout,  # 드롭아웃 비율
        time_bn=args.time_bn,  # 시간 배치 정규화 사용 여부
        channel_bn=args.channel_bn,  # 채널 배치 정규화 사용 여부
        attention=args.attention,  # 어텐션 사용 여부
        avg_channel_attention=args.avg_channel_attention,  # 평균 채널 어텐션 사용 여부
    )

    # 학습 설정
    trainer = Trainer(
        dataset=dataset,
        model=stgan,
        time_step=args.time_steps,
        node_step=args.node_steps,
        batch_size=args.batch_size,
        n_feature=dataset.n_features,
        n_channel=dataset.n_channels,
        is_shuffle=True,
    )

    # 로그 및 모델 저장 디렉토리
    model_dir = f"./bay/checkpoint/pemsbay_{args.missing_type}"
    os.makedirs(model_dir, exist_ok=True)

    # 모델 학습
    print(f"STGAN 모델 학습 시작... (에폭: {args.epochs}, 학습률: {args.lr})")
    trainer.train(model_dir=model_dir, epochs=args.epochs, lr=args.lr, verbose_iter=args.verbose_iter)

    print(f"모델이 {model_dir}에 저장되었습니다.")
    return model_dir


def test_stgan_model(args, data_dir, model_dir):
    """
    학습된 STGAN 모델을 평가합니다.

    Args:
        args: 명령행 인수
        data_dir: 데이터 디렉토리 경로
        model_dir: 모델 디렉토리 경로
    """
    # 데이터셋 로드
    dataset = STGANDataset(data_dir=data_dir)

    # STGAN 모델 생성
    stgan = Generator(
        time_step=args.time_steps,
        node_count=dataset.n_nodes,
        feature_size=dataset.n_features,
        channel_in=dataset.n_channels,
        node_dim=args.node_dim,
        dropout=args.dropout,
        time_bn=args.time_bn,
        channel_bn=args.channel_bn,
        attention=args.attention,
        avg_channel_attention=args.avg_channel_attention,
    )

    # 테스트 설정
    tester = Tester(
        dataset=dataset,
        model=stgan,
        time_step=args.time_steps,
        node_step=args.node_steps,
        batch_size=args.batch_size,
        n_feature=dataset.n_features,
        n_channel=dataset.n_channels,
        is_shuffle=False,
    )

    # 모델 체크포인트 경로 (최신 체크포인트 사용)
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not checkpoints:
        print(f"경고: {model_dir}에 모델 체크포인트가 없습니다.")
        return

    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)

    # 모델 평가
    print(f"STGAN 모델 평가 중... (체크포인트: {checkpoint_path})")
    mse, mae, rmse = tester.test(checkpoint_path=checkpoint_path)

    print("=== 평가 결과 ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 결과 저장
    result_dir = f"./bay/result/pemsbay_{args.missing_type}"
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, "metrics.txt")
    with open(result_file, "w") as f:
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    print(f"평가 결과가 {result_file}에 저장되었습니다.")


def parse_args():
    """
    명령행 인수를 파싱합니다.

    Returns:
        args: 파싱된 인수 객체
    """
    parser = argparse.ArgumentParser(description="STGAN을 이용한 PemsBay 실험")

    # 데이터셋 관련 인수
    parser.add_argument(
        "--missing_type", type=str, default="block", choices=["block", "point"], help="결측치 유형 (block 또는 point)"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--force_regenerate", action="store_true", help="기존 데이터셋이 있어도 다시 생성")

    # 학습 관련 인수
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--time_steps", type=int, default=12, help="시간 창 크기")
    parser.add_argument("--node_steps", type=int, default=5, help="노드 스텝 크기")
    parser.add_argument("--verbose_iter", type=int, default=10, help="학습 상태 출력 간격")

    # 모델 관련 인수
    parser.add_argument("--node_dim", type=int, default=40, help="노드 임베딩 차원")
    parser.add_argument("--dropout", type=float, default=0.1, help="드롭아웃 비율")
    parser.add_argument("--time_bn", action="store_true", help="시간 배치 정규화 사용")
    parser.add_argument("--channel_bn", action="store_true", help="채널 배치 정규화 사용")
    parser.add_argument("--attention", action="store_true", help="어텐션 메커니즘 사용")
    parser.add_argument("--avg_channel_attention", action="store_true", help="평균 채널 어텐션 사용")

    # 테스트 관련 인수
    parser.add_argument("--test_only", action="store_true", help="테스트만 수행")
    parser.add_argument(
        "--model_dir", type=str, default=None, help="테스트할 모델 디렉토리 (test_only가 True일 때 사용)"
    )

    return parser.parse_args()


def main():
    """
    메인 함수: PemsBay 데이터셋에서 STGAN 모델을 학습하고 평가합니다.
    """
    # 인수 파싱
    args = parse_args()

    # 디렉토리 생성
    os.makedirs("./bay/data", exist_ok=True)
    os.makedirs("./bay/checkpoint", exist_ok=True)
    os.makedirs("./bay/result", exist_ok=True)

    # PemsBay 데이터셋 준비
    data_dir = prepare_pemsbay_dataset(args)

    # 테스트만 수행하는 경우
    if args.test_only:
        if args.model_dir is None:
            model_dir = f"./bay/checkpoint/pemsbay_{args.missing_type}"
        else:
            model_dir = args.model_dir

        test_stgan_model(args, data_dir, model_dir)
        return

    # 모델 학습
    model_dir = train_stgan_model(args, data_dir)

    # 모델 평가
    test_stgan_model(args, data_dir, model_dir)


if __name__ == "__main__":
    main()
