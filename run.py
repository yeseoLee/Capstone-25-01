import argparse
from pathlib import Path

import yaml

from .datasets.prepare_datasets import main as prepare_datasets_main
from .experiments import do_evaluate, do_train


def load_config(config_file):
    """YAML 설정 파일 로드"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """설정 딕셔너리를 argparse.Namespace로 변환"""
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    return args


def prepare_datasets():
    """데이터셋 준비"""
    print("데이터셋 준비 중...")

    # 설정 파일 로드
    config = load_config("config/dataset_config.yaml")
    dataset_args = config_to_args(config)

    # 데이터셋 준비
    prepare_datasets_main(dataset_args)


def train_models(args):
    """모델 학습"""
    print("모델 학습 중...")

    # 설정 파일 로드
    config = load_config("config/train_config.yaml")
    train_args = config_to_args(config)

    # 사용자 인자 적용 (명령행 인자가 설정 파일보다 우선함)
    if args.epochs:
        train_args.epochs = args.epochs
    if args.batch_size:
        train_args.batch_size = args.batch_size
    if args.learning_rate:
        train_args.learning_rate = args.learning_rate

    # 학습 실행
    do_train(train_args)


def evaluate_models(args):
    """모델 평가"""
    print("모델 평가 중...")

    # 설정 파일 로드
    config = load_config("config/eval_config.yaml")
    eval_args = config_to_args(config)

    # 사용자 인자 적용
    if args.cpu:
        eval_args.cpu = args.cpu

    # 평가 실행
    do_evaluate(eval_args)


def run_all(args):
    """모든 과정 실행"""
    print("전체 과정 실행 중...")

    # 1. 데이터 생성
    print("\n1. 데이터 생성 중...")
    prepare_datasets()

    # 2. 모델 학습
    print("\n2. 모델 학습 중...")
    train_models(args)

    # 3. 모델 평가
    print("\n3. 모델 평가 중...")
    evaluate_models(args)


def main(args):
    """
    메인 함수
    Args:
        args: 명령행 인자
    """
    # 명령에 따라 적절한 스크립트 실행
    if args.command == "dataset":
        prepare_datasets()
    elif args.command == "train":
        train_models(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    elif args.command == "all":
        run_all(args)
    else:
        print(f"알 수 없는 명령: {args.command}")
        print("사용 가능한 명령: dataset, train, evaluate, all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="교통 데이터 이상치/결측치 보간 프로젝트")

    # 명령 인자
    parser.add_argument(
        "command",
        type=str,
        choices=["dataset", "train", "evaluate", "all"],
        help="실행할 명령 (dataset, train, evaluate, all)",
    )

    # 학습 관련 인자
    parser.add_argument("--epochs", type=int, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, help="학습률")

    # 기타 인자
    parser.add_argument("--cpu", action="store_true", help="CPU 사용 (GPU 대신)")

    args = parser.parse_args()

    # 현재 디렉토리 확인
    current_dir = Path(__file__).parent.absolute()

    # 현재 디렉토리가 프로젝트 루트가 아니라면 경고
    if not (current_dir / "datasets").exists() or not (current_dir / "models").exists():
        print("경고: 프로젝트 루트 디렉토리에서 실행해주세요.")

    main(args)

# 명령어 예시
"""
# 전체 과정 실행
python run.py all

# 또는 단계별 실행
python run.py dataset
python run.py train --epochs 50 --batch_size 64
python run.py evaluate
"""
