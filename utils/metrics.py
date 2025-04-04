import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def masked_mae(y_true, y_pred, mask):
    """
    마스크가 적용된 MAE(Mean Absolute Error) 계산
    Args:
        y_true: 실제값
        y_pred: 예측값
        mask: 평가 마스크 (True인 위치만 계산)
    """
    # 마스크 적용
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # MAE 계산
    return mean_absolute_error(y_true_masked, y_pred_masked)


def masked_rmse(y_true, y_pred, mask):
    """
    마스크가 적용된 RMSE(Root Mean Square Error) 계산
    Args:
        y_true: 실제값
        y_pred: 예측값
        mask: 평가 마스크 (True인 위치만 계산)
    """
    # 마스크 적용
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # RMSE 계산
    return np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))


def masked_mape(y_true, y_pred, mask, epsilon=1e-5):
    """
    마스크가 적용된 MAPE(Mean Absolute Percentage Error) 계산
    Args:
        y_true: 실제값
        y_pred: 예측값
        mask: 평가 마스크 (True인 위치만 계산)
        epsilon: 0으로 나누기 방지를 위한 작은 값
    """
    # 마스크 적용
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # 0으로 나누기 방지
    y_true_safe = np.maximum(np.abs(y_true_masked), epsilon)

    # MAPE 계산
    return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_safe)) * 100


def masked_r2(y_true, y_pred, mask):
    """
    마스크가 적용된 R2 Score 계산
    Args:
        y_true: 실제값
        y_pred: 예측값
        mask: 평가 마스크 (True인 위치만 계산)
    """
    # 마스크 적용
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # R2 계산
    return r2_score(y_true_masked, y_pred_masked)


def evaluate_imputation(y_true, y_pred, missing_mask, outlier_mask):
    """
    결측치 및 이상치 보간 평가
    Args:
        y_true: 실제값
        y_pred: 예측값
        missing_mask: 결측치 마스크
        outlier_mask: 이상치 마스크

    Returns:
        metrics_dict: 평가 지표 딕셔너리
    """
    metrics_dict = {}

    # 결측치 보간 평가
    metrics_dict["missing_mae"] = masked_mae(y_true, y_pred, missing_mask)
    metrics_dict["missing_rmse"] = masked_rmse(y_true, y_pred, missing_mask)
    metrics_dict["missing_mape"] = masked_mape(y_true, y_pred, missing_mask)
    metrics_dict["missing_r2"] = masked_r2(y_true, y_pred, missing_mask)

    # 이상치 보간 평가
    metrics_dict["outlier_mae"] = masked_mae(y_true, y_pred, outlier_mask)
    metrics_dict["outlier_rmse"] = masked_rmse(y_true, y_pred, outlier_mask)
    metrics_dict["outlier_mape"] = masked_mape(y_true, y_pred, outlier_mask)
    metrics_dict["outlier_r2"] = masked_r2(y_true, y_pred, outlier_mask)

    # 전체 보간 평가 (결측치 + 이상치)
    combined_mask = missing_mask | outlier_mask
    metrics_dict["combined_mae"] = masked_mae(y_true, y_pred, combined_mask)
    metrics_dict["combined_rmse"] = masked_rmse(y_true, y_pred, combined_mask)
    metrics_dict["combined_mape"] = masked_mape(y_true, y_pred, combined_mask)
    metrics_dict["combined_r2"] = masked_r2(y_true, y_pred, combined_mask)

    return metrics_dict


def print_metrics(metrics_dict):
    """평가 지표 출력"""
    print("\n=== 결측치 보간 평가 ===")
    print(f"MAE: {metrics_dict['missing_mae']:.4f}")
    print(f"RMSE: {metrics_dict['missing_rmse']:.4f}")
    print(f"MAPE: {metrics_dict['missing_mape']:.4f}%")
    print(f"R2: {metrics_dict['missing_r2']:.4f}")

    print("\n=== 이상치 보간 평가 ===")
    print(f"MAE: {metrics_dict['outlier_mae']:.4f}")
    print(f"RMSE: {metrics_dict['outlier_rmse']:.4f}")
    print(f"MAPE: {metrics_dict['outlier_mape']:.4f}%")
    print(f"R2: {metrics_dict['outlier_r2']:.4f}")

    print("\n=== 종합 보간 평가 ===")
    print(f"MAE: {metrics_dict['combined_mae']:.4f}")
    print(f"RMSE: {metrics_dict['combined_rmse']:.4f}")
    print(f"MAPE: {metrics_dict['combined_mape']:.4f}%")
    print(f"R2: {metrics_dict['combined_r2']:.4f}")
