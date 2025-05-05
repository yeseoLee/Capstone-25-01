import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch


# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    GCNMI,
    GCNODE,
)
from utils import (
    AlternatingPipeline,
    MissingFirstPipeline,
    TempFillPipeline,
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
        missing_model: 결측치 보간 모델
        outlier_model: 이상치 보간 모델
    """
    device = device if torch.cuda.is_available() else "cpu"

    # 결측치 보간 모델 로드
    try:
        gcnmi = GCNMI(in_channels=1).to(device)
        gcnmi.load(os.path.join(model_dir, "gcnmi.pt"), device=device)
        missing_model = gcnmi
    except Exception as e:
        print(f"결측치 모델 로드 중 오류 발생: {e}")
        missing_model = None

    # 이상치 보간 모델 로드
    try:
        gcnode = GCNODE(in_channels=1).to(device)
        gcnode.load(os.path.join(model_dir, "gcnode.pt"), device=device)
        outlier_model = gcnode
    except Exception as e:
        print(f"이상치 모델 로드 중 오류 발생: {e}")
        outlier_model = None

    return missing_model, outlier_model


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
    imputed_data_dict["Outlier_to_Missing"] = final_imputed_data_o2m

    # 성능 평가
    metrics_o2m = evaluate_imputation(original_data, final_imputed_data_o2m, missing_mask, outlier_mask)
    metrics_dict["Outlier_to_Missing"] = metrics_o2m

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
    imputed_data_dict["Missing_to_Outlier"] = final_imputed_data_m2o

    # 성능 평가
    metrics_m2o = evaluate_imputation(original_data, final_imputed_data_m2o, missing_mask, outlier_mask)
    metrics_dict["Missing_to_Outlier"] = metrics_m2o

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


def evaluate_pipelines(test_data, missing_model, outlier_model, result_dir, device="cuda"):
    """
    여러 파이프라인 평가
    Args:
        test_data: 테스트 데이터
        missing_model: 결측치 보간 모델
        outlier_model: 이상치 보간 모델
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

    # 파이프라인 생성
    pipelines = [
        # 1. 결측치 보간 → 이상치 보간 파이프라인
        MissingFirstPipeline(missing_model, outlier_model, device=device),
        # 2. 임시 대체(zero) → 이상치 보간 → 결측치 보간 파이프라인
        TempFillPipeline(missing_model, outlier_model, device=device, temp_value_type="zero"),
        # 2-1. 임시 대체(mean) → 이상치 보간 → 결측치 보간 파이프라인
        TempFillPipeline(missing_model, outlier_model, device=device, temp_value_type="mean"),
        # 2-2. 임시 대체(neighbor) → 이상치 보간 → 결측치 보간 파이프라인
        TempFillPipeline(missing_model, outlier_model, device=device, temp_value_type="neighbor"),
        # 3. 결측치와 이상치 번갈아 보간 파이프라인 (3회 반복)
        AlternatingPipeline(missing_model, outlier_model, device=device, iterations=3),
    ]

    # 결과 저장을 위한 딕셔너리
    imputed_data_dict = {}
    metrics_dict = {}

    # 각 파이프라인별 평가
    for pipeline in pipelines:
        print(f"\n=== {pipeline.name} 평가 ===")

        # 데이터 보간
        imputed_data = pipeline.impute(test_data)
        imputed_data_dict[pipeline.name] = imputed_data

        # 성능 평가
        metrics = evaluate_imputation(original_data, imputed_data, missing_mask, outlier_mask)
        metrics_dict[pipeline.name] = metrics

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
        plt.savefig(
            os.path.join(
                result_dir,
                f"{pipeline.name.replace(' ', '_').replace('(', '').replace(')', '')}_time_series.png",
            )
        )
        plt.close(fig)

    # 파이프라인 비교 시각화
    fig = plot_imputation_comparison(
        original_data,
        mixed_data,
        imputed_data_dict,
        node_idx=0,
        time_range=slice(0, 100),
    )
    plt.savefig(os.path.join(result_dir, "pipelines_comparison.png"))
    plt.close(fig)

    # 성능 비교 시각화
    for metric_name in ["missing_rmse", "outlier_rmse", "combined_rmse"]:
        fig = plot_metrics_comparison(metrics_dict, metric_name=metric_name)
        plt.savefig(os.path.join(result_dir, f"pipelines_{metric_name}_comparison.png"))
        plt.close(fig)

    # 결과 저장
    with open(os.path.join(result_dir, "pipelines_evaluation_results.pkl"), "wb") as f:
        pickle.dump({"imputed_data": imputed_data_dict, "metrics": metrics_dict}, f)

    return metrics_dict


def main(args):
    """
    메인 함수
    Args:
        args: 파라미터
    """
    # GPU 사용 설정
    device = "cpu" if args.cpu else "cuda"
    print(f"Using device: {device if torch.cuda.is_available() else 'cpu'}")

    # 디렉토리 생성
    os.makedirs(args.result_dir, exist_ok=True)

    # 데이터셋 로드
    print("데이터셋 로드 중...")
    data_dict = load_dataset(args.data_path)

    # 데이터 분할
    print("데이터 분할 중...")
    _, _, test_data = split_data(data_dict, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # 모델 로드
    print("모델 로드 중...")
    missing_model, outlier_model = load_models(args.model_dir, device=device)

    # 파이프라인 평가
    print("\n=== 파이프라인 평가 ===")
    pipeline_metrics = evaluate_pipelines(
        test_data,
        missing_model,
        outlier_model,
        os.path.join(args.result_dir, "pipelines"),
        device=device,
    )
    print(pipeline_metrics)

    print("\n평가 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="교통 데이터 이상치/결측치 보간 모델 평가")

    # 데이터 관련 인자
    parser.add_argument("--data_path", type=str, default="datasets/processed/traffic_data.pkl", help="데이터셋 경로")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="학습 데이터 비율")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="검증 데이터 비율")

    # 모델 관련 인자
    parser.add_argument("--model_dir", type=str, default="models/saved/", help="모델 저장 경로")
    parser.add_argument("--result_dir", type=str, default="results/", help="결과 저장 경로")

    # 기타 인자
    parser.add_argument("--cpu", action="store_true", help="CPU 사용 (GPU 대신)")

    # 인자 파싱
    args = parser.parse_args()

    main(args)
