import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch


# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    BIGSAGE,
    GATEDGAT,
    GATODE,
    GCNMI,
    GCNODE,
    GraphSAGEODE,
)
from utils import (
    evaluate_imputation,
    load_dataset,
    plot_imputation_comparison,
    plot_metrics_comparison,
    plot_time_series,
    print_metrics,
    split_data,
)


def load_models(model_dir, device="cuda"):
    """
    학습된 모델 로드
    Args:
        model_dir: 모델 디렉토리
        device: 추론 디바이스

    Returns:
        outlier_models: 이상치 보간 모델 딕셔너리
        missing_models: 결측치 보간 모델 딕셔너리
    """
    outlier_models = {}
    missing_models = {}

    # 이상치 보간 모델 로드
    try:
        gcnode = GCNODE(in_channels=1).to(device)
        gcnode.load(os.path.join(model_dir, "gcnode.pt"), device=device)
        outlier_models["GCNODE"] = gcnode

        gatode = GATODE(in_channels=1).to(device)
        gatode.load(os.path.join(model_dir, "gatode.pt"), device=device)
        outlier_models["GATODE"] = gatode

        graphsageode = GraphSAGEODE(in_channels=1).to(device)
        graphsageode.load(os.path.join(model_dir, "graphsageode.pt"), device=device)
        outlier_models["GraphSAGEODE"] = graphsageode
    except Exception as e:
        print(f"이상치 모델 로드 중 오류 발생: {e}")

    # 결측치 보간 모델 로드
    try:
        gcnmi = GCNMI(in_channels=1).to(device)
        gcnmi.load(os.path.join(model_dir, "gcnmi.pt"), device=device)
        missing_models["GCNMI"] = gcnmi

        bigsage = BIGSAGE(in_channels=1).to(device)
        bigsage.load(os.path.join(model_dir, "bigsage.pt"), device=device)
        missing_models["BIGSAGE"] = bigsage

        gatedgat = GATEDGAT(in_channels=1).to(device)
        gatedgat.load(os.path.join(model_dir, "gatedgat.pt"), device=device)
        missing_models["GATEDGAT"] = gatedgat
    except Exception as e:
        print(f"결측치 모델 로드 중 오류 발생: {e}")

    return outlier_models, missing_models


def evaluate_single_models(test_data, models, result_dir, device="cuda"):
    """
    단일 모델 평가
    Args:
        test_data: 테스트 데이터
        models: 모델 딕셔너리
        result_dir: 결과 저장 경로
        device: 추론 디바이스

    Returns:
        metrics: 성능 지표 딕셔너리
    """
    os.makedirs(result_dir, exist_ok=True)

    # 원본 데이터 및 손상된 데이터
    original_data = test_data["original_data"]
    mixed_data = test_data["mixed_data"]
    missing_mask = test_data["missing_mask"]
    outlier_mask = test_data["outlier_mask"]

    # 결과 저장을 위한 딕셔너리
    imputed_data_dict = {}
    metrics_dict = {}

    # 각 모델별 평가
    for model_name, model in models.items():
        print(f"\n=== {model_name} 모델 평가 ===")

        # 데이터 보간
        imputed_data = model.impute(test_data, device=device)
        imputed_data_dict[model_name] = imputed_data

        # 성능 평가
        metrics = evaluate_imputation(original_data, imputed_data, missing_mask, outlier_mask)
        metrics_dict[model_name] = metrics

        # 성능 출력
        print_metrics(metrics)

        # 시각화
        fig = plot_time_series(
            original_data,
            mixed_data,
            imputed_data,
            node_idx=0,
            time_range=slice(0, 100),
        )
        plt.savefig(os.path.join(result_dir, f"{model_name.lower()}_time_series.png"))
        plt.close(fig)

    # 모델 비교 시각화
    fig = plot_imputation_comparison(
        original_data,
        mixed_data,
        imputed_data_dict,
        node_idx=0,
        time_range=slice(0, 100),
    )
    plt.savefig(os.path.join(result_dir, "model_comparison.png"))
    plt.close(fig)

    # 성능 비교 시각화
    for metric_name in ["missing_rmse", "outlier_rmse", "combined_rmse"]:
        fig = plot_metrics_comparison(metrics_dict, metric_name=metric_name)
        plt.savefig(os.path.join(result_dir, f"{metric_name}_comparison.png"))
        plt.close(fig)

    # 결과 저장
    with open(os.path.join(result_dir, "evaluation_results.pkl"), "wb") as f:
        pickle.dump({"imputed_data": imputed_data_dict, "metrics": metrics_dict}, f)

    return metrics_dict


def evaluate_two_step_imputation(test_data, outlier_models, missing_models, result_dir, device="cuda"):
    """
    2단계 보간 방법 평가 (이상치 → 결측치, 결측치 → 이상치)
    Args:
        test_data: 테스트 데이터
        outlier_models: 이상치 보간 모델 딕셔너리
        missing_models: 결측치 보간 모델 딕셔너리
        result_dir: 결과 저장 경로
        device: 추론 디바이스

    Returns:
        metrics: 성능 지표 딕셔너리
    """
    os.makedirs(result_dir, exist_ok=True)

    # 원본 데이터 및 손상된 데이터
    original_data = test_data["original_data"]
    mixed_data = test_data["mixed_data"]
    missing_mask = test_data["missing_mask"]
    outlier_mask = test_data["outlier_mask"]

    # 결과 저장을 위한 딕셔너리
    imputed_data_dict = {}
    metrics_dict = {}

    # 최고 성능 모델 선택
    best_outlier_model = list(outlier_models.values())[0]  # 기본값
    best_missing_model = list(missing_models.values())[0]  # 기본값

    # 1. 이상치 먼저 보간 후 결측치 보간
    print("\n=== 이상치 → 결측치 보간 ===")

    # 이상치 보간
    outlier_imputed_data = best_outlier_model.impute(test_data, device=device)

    # 이상치 보간 결과로 결측치 보간
    test_data_outlier_imputed = test_data.copy()
    test_data_outlier_imputed["mixed_data"] = outlier_imputed_data

    # 결측치 보간
    final_imputed_data_o2m = best_missing_model.impute(test_data_outlier_imputed, device=device)
    imputed_data_dict["Outlier→Missing"] = final_imputed_data_o2m

    # 성능 평가
    metrics_o2m = evaluate_imputation(original_data, final_imputed_data_o2m, missing_mask, outlier_mask)
    metrics_dict["Outlier→Missing"] = metrics_o2m

    # 성능 출력
    print_metrics(metrics_o2m)

    # 2. 결측치 먼저 보간 후 이상치 보간
    print("\n=== 결측치 → 이상치 보간 ===")

    # 결측치 보간
    missing_imputed_data = best_missing_model.impute(test_data, device=device)

    # 결측치 보간 결과로 이상치 보간
    test_data_missing_imputed = test_data.copy()
    test_data_missing_imputed["mixed_data"] = missing_imputed_data

    # 이상치 보간
    final_imputed_data_m2o = best_outlier_model.impute(test_data_missing_imputed, device=device)
    imputed_data_dict["Missing→Outlier"] = final_imputed_data_m2o

    # 성능 평가
    metrics_m2o = evaluate_imputation(original_data, final_imputed_data_m2o, missing_mask, outlier_mask)
    metrics_dict["Missing→Outlier"] = metrics_m2o

    # 성능 출력
    print_metrics(metrics_m2o)

    # 모델 비교 시각화
    fig = plot_imputation_comparison(
        original_data,
        mixed_data,
        imputed_data_dict,
        node_idx=0,
        time_range=slice(0, 100),
    )
    plt.savefig(os.path.join(result_dir, "two_step_comparison.png"))
    plt.close(fig)

    # 성능 비교 시각화
    for metric_name in ["missing_rmse", "outlier_rmse", "combined_rmse"]:
        fig = plot_metrics_comparison(metrics_dict, metric_name=metric_name)
        plt.savefig(os.path.join(result_dir, f"two_step_{metric_name}_comparison.png"))
        plt.close(fig)

    # 결과 저장
    with open(os.path.join(result_dir, "two_step_results.pkl"), "wb") as f:
        pickle.dump({"imputed_data": imputed_data_dict, "metrics": metrics_dict}, f)

    return metrics_dict


def main(args):
    """
    메인 함수
    Args:
        args: 파라미터
    """
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 로드
    data_dict = load_dataset(args.data_path)

    # 데이터 분할
    train_data, val_data, test_data = split_data(data_dict, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # 모델 로드
    outlier_models, missing_models = load_models(args.model_dir, device=device)

    # 폴더 생성
    os.makedirs(args.result_dir, exist_ok=True)

    # 1. 이상치 보간 모델 평가
    print("\n=== 이상치 보간 모델 평가 ===")
    _ = evaluate_single_models(
        test_data,
        outlier_models,
        os.path.join(args.result_dir, "outlier_models"),
        device=device,
    )

    # 2. 결측치 보간 모델 평가
    print("\n=== 결측치 보간 모델 평가 ===")
    _ = evaluate_single_models(
        test_data,
        missing_models,
        os.path.join(args.result_dir, "missing_models"),
        device=device,
    )

    # 3. 2단계 보간 방법 평가
    print("\n=== 2단계 보간 방법 평가 ===")
    _ = evaluate_two_step_imputation(
        test_data,
        outlier_models,
        missing_models,
        os.path.join(args.result_dir, "two_step"),
        device=device,
    )

    print("\n평가 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="교통 데이터 이상치/결측치 보간 모델 평가")

    # 데이터 관련 인자
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/traffic_data.pkl",
        help="처리된 데이터 경로",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="학습 데이터 비율")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="검증 데이터 비율")

    # 모델 관련 인자
    parser.add_argument("--model_dir", type=str, default="experiments/models", help="모델 저장 디렉토리")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="experiments/results",
        help="결과 저장 디렉토리",
    )
    parser.add_argument("--cpu", action="store_true", help="CPU 사용 (기본: GPU)")

    args = parser.parse_args()

    main(args)
