import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --- Configuration ---
# STGAN_DATA_PATH는 원본 'data.npy'가 있는 폴더입니다.
# create_combined_anomalies.py 내부의 STGAN_DATA_PATH와 일치해야 합니다.

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STGAN_DATA_PATH = Path("./datasets/bay/data")  # 원본 데이터가 있는 경로
OUTPUT_VIZ_DIR_COMBINED = Path("./visualization_results_combined_revised")  # 시각화 결과 저장 폴더
OUTPUT_VIZ_DIR_COMBINED.mkdir(parents=True, exist_ok=True)

# 폰트 설정 (필요시 한글 폰트 지정)
plt.rcParams["font.family"] = "AppleGothic"  # macOS
# plt.rcParams['font.family'] = 'NanumGothic' # Windows/Linux with Nanum font
plt.rcParams["axes.unicode_minus"] = False


# --- Helper Functions (이전 스크립트에서 가져오거나 약간 수정) ---
def load_original_pems_data(original_data_file_path: Path):
    """원본 PEMS 데이터 로드"""
    if not original_data_file_path.exists():
        raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {original_data_file_path}")
    print(f"원본 데이터 로드 중: {original_data_file_path}")
    data = np.load(original_data_file_path)
    if np.isnan(data).any():
        print("경고: 원본 데이터에 NaN 값이 있습니다. 분석의 일관성을 위해 0으로 대체합니다.")
        data = np.nan_to_num(data, nan=0.0)
    return data


def load_combined_anomalies_pems_data(synthetic_folder_path: Path):
    """결합된 이상치/결측치 합성 데이터 및 마스크 로드"""
    synthetic_folder_path = Path(synthetic_folder_path)
    data_file = synthetic_folder_path / "data.npy"
    outlier_mask_file = synthetic_folder_path / "outlier_mask.npy"
    missing_mask_file = synthetic_folder_path / "missing_mask.npy"

    if not data_file.exists():
        raise FileNotFoundError(f"합성 데이터 파일을 찾을 수 없습니다: {data_file}")
    if not outlier_mask_file.exists():
        raise FileNotFoundError(f"이상치 마스크 파일을 찾을 수 없습니다: {outlier_mask_file}")
    if not missing_mask_file.exists():
        raise FileNotFoundError(f"결측치 마스크 파일을 찾을 수 없습니다: {missing_mask_file}")

    print(f"합성 데이터 로드 중: {data_file}")
    synthetic_data = np.load(data_file)  # 결측은 NaN, 이상치는 값으로 존재

    print(f"이상치 마스크 로드 중: {outlier_mask_file}")
    # create_combined_anomalies.py에서 outlier_mask.npy는 True가 정상, False가 이상치를 의미
    raw_outlier_mask = np.load(outlier_mask_file)
    # 시각화 통일성을 위해 True가 이상치를 의미하도록 변환
    is_outlier_mask = ~raw_outlier_mask

    print(f"결측치 마스크 로드 중: {missing_mask_file}")
    # create_combined_anomalies.py에서 missing_mask.npy는 True가 정상, False가 결측을 의미
    raw_missing_mask = np.load(missing_mask_file)
    # 시각화 통일성을 위해 True가 결측을 의미하도록 변환
    is_missing_mask = ~raw_missing_mask

    return synthetic_data, is_outlier_mask, is_missing_mask


def print_basic_stats_combined(data, name="데이터", outlier_mask=None, missing_mask=None):
    """데이터의 기본 통계량 및 이상치/결측치 비율 출력 (결합된 경우)"""
    print(f"\n--- {name} 기본 통계 ---")
    data_for_stats = data.copy()

    if np.isnan(data_for_stats).any():
        print(f"  형태: {data_for_stats.shape}")
        print(f"  평균 (NaN 제외): {np.nanmean(data_for_stats):.4f}")
        print(f"  표준편차 (NaN 제외): {np.nanstd(data_for_stats):.4f}")
        print(f"  최소값 (NaN 제외): {np.nanmin(data_for_stats):.4f}")
        print(f"  최대값 (NaN 제외): {np.nanmax(data_for_stats):.4f}")
    else:
        print(f"  형태: {data_for_stats.shape}")
        print(f"  평균: {np.mean(data_for_stats):.4f}")
        print(f"  표준편차: {np.std(data_for_stats):.4f}")
        print(f"  최소값: {np.min(data_for_stats):.4f}")
        print(f"  최대값: {np.max(data_for_stats):.4f}")

    if outlier_mask is not None:
        outlier_ratio = np.mean(outlier_mask)
        print(f"  이상치 비율 (전체 데이터 포인트 중): {outlier_ratio:.4f} ({outlier_ratio * 100:.2f}%)")
        # data[outlier_mask] can contain NaNs if an outlier position also became NaN by mistake (should not happen with current create_combined_anomalies logic)
        outlier_values = data[outlier_mask & ~np.isnan(data)]  # Consider only actual outlier values
        if outlier_values.size > 0:
            print(f"  이상치 값 평균: {np.mean(outlier_values):.4f}")
            print(f"  이상치 값 표준편차: {np.std(outlier_values):.4f}")
        elif data[outlier_mask].size > 0:  # Outlier mask is true, but values are NaN (should not happen)
            print("  이상치로 표시되었으나 해당 위치 값이 NaN입니다.")
        else:
            print("  이상치 값 없음")

    if missing_mask is not None:
        missing_ratio = np.mean(missing_mask)
        print(f"  결측치 마스크 비율 (전체 데이터 포인트 중): {missing_ratio:.4f} ({missing_ratio * 100:.2f}%)")
        actual_nan_ratio = np.sum(np.isnan(data)) / data.size
        print(f"  실제 NaN 비율 (데이터 내): {actual_nan_ratio:.4f} ({actual_nan_ratio * 100:.2f}%)")
        # Check consistency
        if not np.all(missing_mask[np.isnan(data)]):
            print("  경고: 데이터 내 NaN 위치와 결측치 마스크가 불일치하는 부분이 있습니다.")
        if not np.all(~missing_mask[~np.isnan(data)]):
            print("  경고: 데이터 내 non-NaN 위치와 결측치 마스크가 불일치하는 부분이 있습니다.")

    if outlier_mask is not None and missing_mask is not None:
        combined_anomaly_mask = outlier_mask | missing_mask
        total_anomaly_ratio = np.mean(combined_anomaly_mask)
        print(f"  총 비정상(이상치 마스크 True 또는 결측치 마스크 True) 비율: {total_anomaly_ratio:.4f} ({total_anomaly_ratio * 100:.2f}%)")


# --- 시각화 함수 (수정됨) ---
def plot_value_distribution(original_data, synthetic_data_raw, data_label, feature_index=0, channel_index=0, sample_size=50000, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """값 분포 비교 (히스토그램 및 KDE) - 합성 데이터의 결측은 0으로 대체"""
    plt.figure(figsize=(12, 6))

    if original_data.ndim == 4 and synthetic_data_raw.ndim == 4:
        orig_flat = original_data[:, :, feature_index, channel_index].flatten()
        synth_flat_raw = synthetic_data_raw[:, :, feature_index, channel_index].flatten()
        plot_title = f"값 분포 (피처 {feature_index}, 채널 {channel_index})"
    else:
        orig_flat = original_data.flatten()
        synth_flat_raw = synthetic_data_raw.flatten()
        plot_title = "전체 값 분포"

    # 원본 데이터는 로드 시 NaN이 0으로 처리됨 (load_original_pems_data 참조)
    orig_flat_processed = orig_flat
    # 합성 데이터의 NaN(결측)을 0으로 대체
    synth_flat_imputed = np.nan_to_num(synth_flat_raw, nan=0.0)

    if len(orig_flat_processed) > sample_size:
        orig_flat_processed = np.random.choice(orig_flat_processed, sample_size, replace=False)
    if len(synth_flat_imputed) > sample_size:
        synth_flat_imputed = np.random.choice(synth_flat_imputed, sample_size, replace=False)

    sns.histplot(orig_flat_processed, color="blue", label="원본", kde=True, stat="density", element="step", alpha=0.7)
    sns.histplot(synth_flat_imputed, color="red", label=f"{data_label} (결측=0)", kde=True, stat="density", element="step", alpha=0.7)

    plt.title(plot_title)
    plt.xlabel("값")
    plt.ylabel("밀도")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}value_distribution_feat{feature_index}_ch{channel_index}.png")
    plt.close()
    print(f"값 분포 비교 그래프 저장 완료: {output_dir / f'{prefix}value_distribution_feat{feature_index}_ch{channel_index}.png'}")


def plot_time_series_comparison_combined(
    original_data, synthetic_data_raw, data_label, sensor_index=0, feature_index=0, channel_index=0, time_points=500, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""
):
    """특정 센서의 시계열 비교 (합성 데이터의 결측은 0으로 대체)"""
    plt.figure(figsize=(18, 8))

    orig_ts = original_data[:time_points, sensor_index, feature_index, channel_index]
    # 합성 데이터 로드 및 결측치 0으로 대체
    synth_ts_values = synthetic_data_raw[:time_points, sensor_index, feature_index, channel_index]
    synth_ts_imputed = np.nan_to_num(synth_ts_values, nan=0.0)

    time_axis = np.arange(len(orig_ts))

    # 원본 데이터 플롯
    plt.plot(time_axis, orig_ts, label="원본", color="blue", alpha=0.7, zorder=1)

    # 수정된 합성 데이터 플롯
    plt.plot(time_axis, synth_ts_imputed, label=f"{data_label} (결측=0)", color="red", alpha=0.7, linestyle="--", zorder=2)

    plt.title(f"시계열 비교 (센서 {sensor_index}, 결측=0)")
    plt.xlabel("시간 스텝")
    plt.ylabel("값")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}timeseries_sensor{sensor_index}_feat{feature_index}_ch{channel_index}.png")
    plt.close()
    print(f"시계열 비교 그래프 저장 완료: {output_dir / f'{prefix}timeseries_sensor{sensor_index}_feat{feature_index}_ch{channel_index}.png'}")


def plot_anomaly_mask_heatmap(mask, anomaly_type_label, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """이상치 또는 결측치 마스크를 시간-센서 히트맵으로 시각화 (이 함수는 변경 없음)"""
    if mask.ndim == 4:
        mask_2d = mask[:, :, 0, 0]
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        print(f"지원하지 않는 마스크 차원({mask.ndim})입니다. 2D 또는 4D 마스크가 필요합니다.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(mask_2d.T, cmap="viridis_r", cbar=True)
    plt.title(f"{anomaly_type_label} 발생 분포 (시간 vs 센서)")
    plt.xlabel("시간 스텝")
    plt.ylabel("센서 ID")
    plt.tight_layout()
    # Corrected savefig path to include prefix
    plt.savefig(output_dir / f"{prefix}_distribution_heatmap.png")
    plt.close()
    print(f"{anomaly_type_label} 분포 히트맵 저장 완료: {output_dir / f'{prefix}_distribution_heatmap.png'}")


def plot_pca_comparison_combined(original_data, synthetic_data_raw, data_label, sample_size=2000, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """PCA를 이용한 2D 시각화 비교 (합성 데이터의 결측은 0으로 대체)"""
    num_time_orig, num_nodes_orig, _, _ = original_data.shape
    num_time_synth, num_nodes_synth, _, _ = synthetic_data_raw.shape

    # 원본 데이터는 로드 시 NaN이 0으로 처리됨
    orig_reshaped = original_data.reshape(num_time_orig * num_nodes_orig, -1)
    # 합성 데이터는 NaN을 0으로 대체
    synth_reshaped_imputed = np.nan_to_num(synthetic_data_raw.reshape(num_time_synth * num_nodes_synth, -1), nan=0.0)

    # 샘플링
    indices_orig = np.random.choice(orig_reshaped.shape[0], min(orig_reshaped.shape[0], sample_size), replace=False)
    orig_sampled = orig_reshaped[indices_orig]

    indices_synth = np.random.choice(synth_reshaped_imputed.shape[0], min(synth_reshaped_imputed.shape[0], sample_size), replace=False)
    synth_sampled = synth_reshaped_imputed[indices_synth]

    # PCA 적용 (원본 데이터 기준으로 학습)
    pca = PCA(n_components=2)
    pca.fit(orig_sampled)
    orig_pca = pca.transform(orig_sampled)
    synth_pca = pca.transform(synth_sampled)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.5, label="원본", s=10, color="blue")
    plt.title("PCA of Original Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label=f"{data_label} (결측=0)", color="red", s=10)
    plt.title(f"PCA of {data_label} (결측=0)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.suptitle("PCA 비교", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"{prefix}pca_comparison_combined.png")
    plt.close()
    print(f"PCA 비교 (결합) 그래프 저장 완료: {output_dir / f'{prefix}pca_comparison_combined.png'}")


def plot_tsne_comparison_combined(original_data, synthetic_data_raw, data_label, sample_size=1000, perplexity=30, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """t-SNE를 이용한 2D 시각화 비교 (합성 데이터의 결측은 0으로 대체)"""
    num_time_orig, num_nodes_orig, _, _ = original_data.shape
    num_time_synth, num_nodes_synth, _, _ = synthetic_data_raw.shape

    orig_reshaped = original_data.reshape(num_time_orig * num_nodes_orig, -1)
    synth_reshaped_imputed = np.nan_to_num(synthetic_data_raw.reshape(num_time_synth * num_nodes_synth, -1), nan=0.0)

    indices_orig = np.random.choice(orig_reshaped.shape[0], min(orig_reshaped.shape[0], sample_size), replace=False)
    orig_sampled = orig_reshaped[indices_orig]

    indices_synth = np.random.choice(synth_reshaped_imputed.shape[0], min(synth_reshaped_imputed.shape[0], sample_size), replace=False)
    synth_sampled = synth_reshaped_imputed[indices_synth]

    combined_data_sampled = np.vstack((orig_sampled, synth_sampled))

    print(f"t-SNE 계산 중... (샘플: {combined_data_sampled.shape[0]}, 시간 소요)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300, init="pca", learning_rate="auto")  # Explicitly set n_iter and init for reproducibility/control.
    combined_tsne = tsne.fit_transform(combined_data_sampled)

    orig_tsne = combined_tsne[: len(orig_sampled)]
    synth_tsne = combined_tsne[len(orig_sampled) :]

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_tsne[:, 0], orig_tsne[:, 1], alpha=0.5, label="원본", s=10, color="blue")
    plt.title("t-SNE of Original Data")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.scatter(synth_tsne[:, 0], synth_tsne[:, 1], alpha=0.5, label=f"{data_label} (결측=0)", color="red", s=10)
    plt.title(f"t-SNE of {data_label} (결측=0)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.suptitle("t-SNE 비교", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"{prefix}tsne_comparison_combined.png")
    plt.close()
    print(f"t-SNE 비교 (결합) 그래프 저장 완료: {output_dir / f'{prefix}tsne_comparison_combined.png'}")


def plot_spatial_distribution_at_time(original_data, synthetic_data_raw, data_label, timestamp_idx=100, feature_index=0, channel_index=0, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """특정 시점에서의 센서 값 공간 분포 비교 (합성 데이터의 결측은 0으로 대체)"""
    orig_spatial = original_data[timestamp_idx, :, feature_index, channel_index]
    synth_spatial_values = synthetic_data_raw[timestamp_idx, :, feature_index, channel_index]
    synth_spatial_imputed = np.nan_to_num(synth_spatial_values, nan=0.0)

    num_sensors = original_data.shape[1]
    sensor_ids = np.arange(num_sensors)

    # Calculate min/max based on processed data to ensure consistent y-limits
    all_values_for_ylim = np.concatenate([orig_spatial, synth_spatial_imputed])
    min_val = np.min(all_values_for_ylim) if all_values_for_ylim.size > 0 else 0
    max_val = np.max(all_values_for_ylim) if all_values_for_ylim.size > 0 else 1
    if min_val == max_val:  # Avoid flat line if all values are same
        min_val -= 0.1 * abs(min_val) if min_val != 0 else 0.1
        max_val += 0.1 * abs(max_val) if max_val != 0 else 0.1
        if min_val == max_val:  # still same (e.g. both were 0)
            min_val -= 0.5
            max_val += 0.5

    plt.figure(figsize=(18, 10))

    # 원본 데이터 공간 분포
    plt.subplot(2, 1, 1)
    plt.bar(sensor_ids, orig_spatial, color="skyblue", label="원본")
    plt.title(f"원본 데이터 공간 분포 (시간: {timestamp_idx})")
    plt.xlabel("센서 ID")
    plt.ylabel("값")
    plt.ylim(min_val, max_val)
    plt.xticks(ticks=sensor_ids[:: max(1, num_sensors // 30)], labels=sensor_ids[:: max(1, num_sensors // 30)], rotation=45, ha="right", fontsize=8)
    plt.grid(True, axis="y", alpha=0.5)
    plt.legend()

    # 합성 데이터 공간 분포 (결측=0)
    plt.subplot(2, 1, 2)
    plt.bar(sensor_ids, synth_spatial_imputed, color="lightcoral", label=f"{data_label} (결측=0)")
    plt.title(f"{data_label} 공간 분포 (시간: {timestamp_idx}, 결측=0)")
    plt.xlabel("센서 ID")
    plt.ylabel("값")
    plt.ylim(min_val, max_val)
    plt.xticks(ticks=sensor_ids[:: max(1, num_sensors // 30)], labels=sensor_ids[:: max(1, num_sensors // 30)], rotation=45, ha="right", fontsize=8)
    plt.grid(True, axis="y", alpha=0.5)
    plt.legend()

    plt.suptitle("공간 분포 비교", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"{prefix}spatial_distribution_time{timestamp_idx}_combined.png")
    plt.close()
    print(f"공간 분포 (결합) 비교 그래프 저장 완료: {output_dir / f'{prefix}spatial_distribution_time{timestamp_idx}_combined.png'}")


# --- Main Execution ---
if __name__ == "__main__":
    original_data_file = STGAN_DATA_PATH / "data.npy"
    # !!! 아래 경로를 실제 생성된 폴더명으로 수정해야 합니다 !!!
    SYNTHETIC_COMBINED_FOLDER_PATH = Path("./datasets/bay/outliers/contextual_ratio0.05_interval0.00-1.00_nodes0.10")  # Placeholder, update this!
    # SYNTHETIC_COMBINED_FOLDER_PATH = Path("./datasets/bay/combined_test/combined_point_dev0.20-0.50_p0.0100_fault0.0015_noise0.0500_interval0.00-1.00") # Example path

    SYNTHETIC_DATA_NAME = "합성"  # 그래프 제목 등에 사용될 이름 (간결하게)

    try:
        if not SYNTHETIC_COMBINED_FOLDER_PATH.exists() or not SYNTHETIC_COMBINED_FOLDER_PATH.is_dir():
            # Try to find a suitable directory if the placeholder is used, or if the path is wrong
            if "block_dev" in str(SYNTHETIC_COMBINED_FOLDER_PATH) or "combined_point" in str(SYNTHETIC_COMBINED_FOLDER_PATH):  # Default name used
                print(f"경고: 지정된 합성 데이터 폴더 '{SYNTHETIC_COMBINED_FOLDER_PATH}'를 찾을 수 없습니다.")
                # Attempt to find a subdirectory in ./datasets/bay/ (common parent for outliers or combined_test)
                potential_parent_dirs = [Path("./datasets/bay/outliers"), Path("./datasets/bay/combined_test")]
                found_dir = None
                for p_dir in potential_parent_dirs:
                    if p_dir.exists() and p_dir.is_dir():
                        subdirs = [d for d in p_dir.iterdir() if d.is_dir()]
                        if subdirs:
                            SYNTHETIC_COMBINED_FOLDER_PATH = subdirs[0]  # Take the first one found
                            print(f"대신 '{SYNTHETIC_COMBINED_FOLDER_PATH}' 폴더를 사용합니다. 필요시 경로를 수정하세요.")
                            found_dir = True
                            break
                if not found_dir:
                    raise FileNotFoundError(f"지정된 합성 데이터 폴더 '{SYNTHETIC_COMBINED_FOLDER_PATH}'를 찾을 수 없습니다. `create_combined_anomalies.py` 실행 후 생성된 폴더 경로로 수정해주세요.")
            else:  # User likely provided a path, but it's wrong
                raise FileNotFoundError(f"지정된 합성 데이터 폴더 '{SYNTHETIC_COMBINED_FOLDER_PATH}'를 찾을 수 없습니다. 경로를 확인해주세요.")

        original_pems_data = load_original_pems_data(original_data_file)
        # Pass the raw synthetic data to plotting functions, they will handle np.nan_to_num
        synthetic_pems_data_raw, is_outlier_mask, is_missing_mask = load_combined_anomalies_pems_data(SYNTHETIC_COMBINED_FOLDER_PATH)

        viz_prefix = SYNTHETIC_COMBINED_FOLDER_PATH.name + "_"

        print_basic_stats_combined(original_pems_data, name="원본 데이터")
        # For stats, we pass the raw data and masks to get accurate info on NaNs vs masks
        print_basic_stats_combined(synthetic_pems_data_raw, name=f"{SYNTHETIC_DATA_NAME} 데이터 (Raw)", outlier_mask=is_outlier_mask, missing_mask=is_missing_mask)

        # Create a version of synthetic data with NaNs imputed to 0 for general use if needed,
        # though most plotting functions now do this internally.
        # synthetic_pems_data_imputed = np.nan_to_num(synthetic_pems_data_raw, nan=0.0)

        # 1. 값 분포 비교
        plot_value_distribution(original_pems_data, synthetic_pems_data_raw, data_label=SYNTHETIC_DATA_NAME, feature_index=0, channel_index=0, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 2. 특정 센서 시계열 비교
        plot_time_series_comparison_combined(
            original_pems_data,
            synthetic_pems_data_raw,  # Pass raw, imputation inside function
            data_label=SYNTHETIC_DATA_NAME,
            sensor_index=10,
            feature_index=0,
            channel_index=0,
            time_points=700,
            output_dir=OUTPUT_VIZ_DIR_COMBINED,
            prefix=viz_prefix,
        )

        # 3. 이상치 발생 분포 히트맵 (마스크 사용)
        plot_anomaly_mask_heatmap(is_outlier_mask, anomaly_type_label="이상치(Outlier) 마스크", output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 4. 결측치 발생 분포 히트맵 (마스크 사용)
        plot_anomaly_mask_heatmap(is_missing_mask, anomaly_type_label="결측치(Missing) 마스크", output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 5. PCA 비교
        plot_pca_comparison_combined(original_pems_data, synthetic_pems_data_raw, data_label=SYNTHETIC_DATA_NAME, sample_size=2000, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 6. t-SNE 비교
        plot_tsne_comparison_combined(
            original_pems_data,
            synthetic_pems_data_raw,
            data_label=SYNTHETIC_DATA_NAME,
            sample_size=1000,  # Keep sample size small for t-SNE speed
            perplexity=30,
            output_dir=OUTPUT_VIZ_DIR_COMBINED,
            prefix=viz_prefix,
        )

        # 7. 특정 시점 공간 분포 비교
        plot_spatial_distribution_at_time(
            original_pems_data,
            synthetic_pems_data_raw,
            data_label=SYNTHETIC_DATA_NAME,
            timestamp_idx=150,
            feature_index=0,
            channel_index=0,
            output_dir=OUTPUT_VIZ_DIR_COMBINED,
            prefix=viz_prefix,
        )

        print(f"\n모든 시각화가 완료되었습니다. 결과는 '{OUTPUT_VIZ_DIR_COMBINED}' 폴더에서 확인하세요.")

    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("지정된 경로에 원본 데이터 파일 또는 합성 데이터 폴더가 있는지, 파일명이 정확한지 확인해주세요.")
        print(f"  - 원본 데이터 파일 예상 경로: {original_data_file}")
        print(f"  - 확인된/시도된 합성 데이터 폴더 경로: {SYNTHETIC_COMBINED_FOLDER_PATH}")
        print("  - `SYNTHETIC_COMBINED_FOLDER_PATH`를 `create_combined_anomalies.py` 실행 후 생성된 실제 폴더 경로로 수정하십시오.")
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
