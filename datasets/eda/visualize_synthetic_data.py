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
OUTPUT_VIZ_DIR_COMBINED = Path("./visualization_results_combined")  # 시각화 결과 저장 폴더
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

    # 중요: create_combined_anomalies.py의 로직상,
    # outlier_mask는 이미 missing_mask를 고려하여 결측 위치에는 이상치가 없도록 처리됨.
    # 즉, is_outlier_mask가 True인 곳은 실제로 이상치 값이며, NaN이 아님.
    # is_missing_mask가 True인 곳은 synthetic_data에서 NaN 값을 가짐.

    return synthetic_data, is_outlier_mask, is_missing_mask


def print_basic_stats_combined(data, name="데이터", outlier_mask=None, missing_mask=None):
    """데이터의 기본 통계량 및 이상치/결측치 비율 출력 (결합된 경우)"""
    print(f"\n--- {name} 기본 통계 ---")
    data_for_stats = data.copy()  # 원본 데이터 보존

    if np.isnan(data_for_stats).any():  # 결측(NaN)이 포함된 데이터
        print(f"  형태: {data_for_stats.shape}")
        print(f"  평균 (NaN 제외): {np.nanmean(data_for_stats):.4f}")
        print(f"  표준편차 (NaN 제외): {np.nanstd(data_for_stats):.4f}")
        print(f"  최소값 (NaN 제외): {np.nanmin(data_for_stats):.4f}")
        print(f"  최대값 (NaN 제외): {np.nanmax(data_for_stats):.4f}")
    else:  # NaN이 없는 데이터 (예: 원본 데이터)
        print(f"  형태: {data_for_stats.shape}")
        print(f"  평균: {np.mean(data_for_stats):.4f}")
        print(f"  표준편차: {np.std(data_for_stats):.4f}")
        print(f"  최소값: {np.min(data_for_stats):.4f}")
        print(f"  최대값: {np.max(data_for_stats):.4f}")

    if outlier_mask is not None:
        # is_outlier_mask는 True가 이상치
        outlier_ratio = np.mean(outlier_mask)
        print(f"  이상치 비율 (전체 데이터 포인트 중): {outlier_ratio:.4f} ({outlier_ratio * 100:.2f}%)")
        outlier_values = data[outlier_mask]  # is_outlier_mask가 True인 곳의 값
        if outlier_values.size > 0 and not np.all(np.isnan(outlier_values)):  # 이상치 값들이 NaN이 아닌 경우
            print(f"  이상치 값 평균: {np.nanmean(outlier_values):.4f}")
            print(f"  이상치 값 표준편차: {np.nanstd(outlier_values):.4f}")
        elif outlier_values.size == 0:
            print("  이상치 값 없음")

    if missing_mask is not None:
        # is_missing_mask는 True가 결측
        missing_ratio = np.mean(missing_mask)
        print(f"  결측치 비율 (전체 데이터 포인트 중): {missing_ratio:.4f} ({missing_ratio * 100:.2f}%)")
        # 실제 NaN 개수와도 비교 가능
        actual_nan_ratio = np.sum(np.isnan(data)) / data.size
        print(f"  실제 NaN 비율 (데이터 내): {actual_nan_ratio:.4f} ({actual_nan_ratio * 100:.2f}%)")

    if outlier_mask is not None and missing_mask is not None:
        # True가 비정상 (이상치 또는 결측)
        combined_anomaly_mask = outlier_mask | missing_mask
        total_anomaly_ratio = np.mean(combined_anomaly_mask)
        print(f"  총 비정상(이상치+결측치) 비율: {total_anomaly_ratio:.4f} ({total_anomaly_ratio * 100:.2f}%)")


# --- 시각화 함수 (이전 스크립트에서 가져와서 수정) ---
def plot_value_distribution(original_data, synthetic_data, data_label, feature_index=0, channel_index=0, sample_size=50000, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """값 분포 비교 (히스토그램 및 KDE) - 이전과 동일하게 사용 가능, synthetic_data의 NaN은 자동 제외됨"""
    plt.figure(figsize=(12, 6))

    if original_data.ndim == 4 and synthetic_data.ndim == 4:
        orig_flat = original_data[:, :, feature_index, channel_index].flatten()
        synth_flat = synthetic_data[:, :, feature_index, channel_index].flatten()
        plot_title = f"값 분포 (피처 {feature_index}, 채널 {channel_index})"
    else:
        orig_flat = original_data.flatten()
        synth_flat = synthetic_data.flatten()
        plot_title = "전체 값 분포"

    orig_flat_no_nan = orig_flat[~np.isnan(orig_flat)]
    synth_flat_no_nan = synth_flat[~np.isnan(synth_flat)]  # 합성 데이터의 NaN(결측) 제외

    if len(orig_flat_no_nan) > sample_size:
        orig_flat_no_nan = np.random.choice(orig_flat_no_nan, sample_size, replace=False)
    if len(synth_flat_no_nan) > sample_size:
        synth_flat_no_nan = np.random.choice(synth_flat_no_nan, sample_size, replace=False)

    sns.histplot(orig_flat_no_nan, color="blue", label="원본 데이터", kde=True, stat="density", element="step", alpha=0.7)
    sns.histplot(synth_flat_no_nan, color="red", label=data_label, kde=True, stat="density", element="step", alpha=0.7)

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
    original_data, synthetic_data, is_outlier_mask, is_missing_mask, data_label, sensor_index=0, feature_index=0, channel_index=0, time_points=500, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""
):
    """특정 센서의 시계열 비교 (이상치 및 결측치 구분 강조)"""
    plt.figure(figsize=(18, 8))

    orig_ts = original_data[:time_points, sensor_index, feature_index, channel_index]
    synth_ts_raw = synthetic_data[:time_points, sensor_index, feature_index, channel_index]  # NaN 포함 가능

    outlier_flags = is_outlier_mask[:time_points, sensor_index, feature_index, channel_index]  # True가 이상치
    missing_flags = is_missing_mask[:time_points, sensor_index, feature_index, channel_index]  # True가 결측

    time_axis = np.arange(len(orig_ts))

    # 원본 데이터 플롯
    plt.plot(time_axis, orig_ts, label="원본 데이터", color="blue", alpha=0.6, zorder=1)

    # 합성 데이터 플롯 (NaN은 끊어져서 그려짐)
    plt.plot(time_axis, synth_ts_raw, label=data_label, color="red", alpha=0.7, linestyle="--", zorder=2)

    # 이상치 위치 표시 (보라색 X)
    # 이상치는 synthetic_data에 실제 값이 있으므로 그 값을 사용
    outlier_indices_ts = time_axis[outlier_flags]
    if outlier_indices_ts.size > 0:
        plt.scatter(outlier_indices_ts, synth_ts_raw[outlier_flags], color="purple", marker="x", s=80, label=f"이상치 ({data_label})", zorder=3)

    # 결측치 위치 표시 (주황색 O, 원본 데이터 위치에 표시)
    # 결측치는 synthetic_data에서 NaN이므로, 원본 데이터의 y값을 참조하여 표시
    missing_indices_ts = time_axis[missing_flags]
    if missing_indices_ts.size > 0:
        plt.scatter(missing_indices_ts, orig_ts[missing_flags], edgecolor="orange", facecolor="none", marker="o", s=80, label=f"결측치 ({data_label})", zorder=4)

    plt.title(f"시계열 비교 (센서 {sensor_index}, 피처 {feature_index}, 채널 {channel_index})")
    plt.xlabel("시간 스텝")
    plt.ylabel("값")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"timeseries_sensor{sensor_index}_feat{feature_index}_ch{channel_index}.png")
    plt.close()
    print(f"시계열 비교 그래프 저장 완료: {output_dir / f'timeseries_sensor{sensor_index}_feat{feature_index}_ch{channel_index}.png'}")


def plot_anomaly_mask_heatmap(mask, anomaly_type_label, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """이상치 또는 결측치 마스크를 시간-센서 히트맵으로 시각화"""
    if mask.ndim == 4:
        # 첫번째 피처, 첫번째 채널의 마스크 사용 또는 any 연산
        mask_2d = mask[:, :, 0, 0]
        # mask_2d = np.any(mask, axis=(2,3)) # 모든 피처/채널 중 하나라도 해당되면 True
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        print(f"지원하지 않는 마스크 차원({mask.ndim})입니다. 2D 또는 4D 마스크가 필요합니다.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(mask_2d.T, cmap="viridis_r", cbar=True)  # Transpose for (sensor, time)
    plt.title(f"{anomaly_type_label} 발생 분포 (시간 vs 센서)")
    plt.xlabel("시간 스텝")
    plt.ylabel("센서 ID")
    plt.tight_layout()
    plt.savefig(output_dir / f"{anomaly_type_label.lower()}_distribution_heatmap.png")
    plt.close()
    print(f"{anomaly_type_label} 분포 히트맵 저장 완료: {output_dir / f'{prefix}{anomaly_type_label.lower()}_distribution_heatmap.png'}")


def plot_pca_comparison_combined(original_data, synthetic_data, is_outlier_mask, is_missing_mask, data_label, sample_size=2000, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """PCA를 이용한 2D 시각화 비교 (정상, 이상치, 결측치 구분)"""
    num_time_orig, num_nodes_orig, _, _ = original_data.shape
    num_time_synth, num_nodes_synth, _, _ = synthetic_data.shape

    orig_reshaped = original_data.reshape(num_time_orig * num_nodes_orig, -1)
    # 합성 데이터는 NaN을 0으로 대체하여 PCA 수행 (다른 방법도 고려 가능)
    synth_reshaped_no_nan = np.nan_to_num(synthetic_data.reshape(num_time_synth * num_nodes_synth, -1), nan=0.0)

    # 샘플 레벨 마스크 (각 (시간,노드) 샘플이 이상치인지, 결측인지)
    # (시간, 노드, 피처, 채널) -> (시간*노드, 피처*채널) -> (시간*노드)
    # 어떤 피처/채널이든 하나라도 이상치/결측이면 해당 샘플을 이상치/결측로 간주
    sample_is_outlier = np.any(is_outlier_mask.reshape(num_time_synth * num_nodes_synth, -1), axis=1)
    sample_is_missing = np.any(is_missing_mask.reshape(num_time_synth * num_nodes_synth, -1), axis=1)

    # 샘플링 (PCA 계산 시간 단축)
    indices_orig = np.random.choice(orig_reshaped.shape[0], min(orig_reshaped.shape[0], sample_size), replace=False)
    orig_sampled = orig_reshaped[indices_orig]

    indices_synth = np.random.choice(synth_reshaped_no_nan.shape[0], min(synth_reshaped_no_nan.shape[0], sample_size), replace=False)
    synth_sampled = synth_reshaped_no_nan[indices_synth]
    sample_is_outlier_sampled = sample_is_outlier[indices_synth]
    sample_is_missing_sampled = sample_is_missing[indices_synth]

    # PCA 적용 (원본 데이터 기준으로 학습)
    pca = PCA(n_components=2)
    pca.fit(orig_sampled)  # 원본 데이터로 PCA 모델 학습
    orig_pca = pca.transform(orig_sampled)
    synth_pca = pca.transform(synth_sampled)  # 학습된 모델로 합성 데이터 변환

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.5, label="원본 데이터", s=10)
    plt.title("PCA of Original Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    # 정상, 이상치, 결측치 포인트 분리
    normal_flags_sampled = ~sample_is_outlier_sampled & ~sample_is_missing_sampled

    plt.scatter(synth_pca[normal_flags_sampled, 0], synth_pca[normal_flags_sampled, 1], alpha=0.4, label=f"정상 ({data_label})", color="green", s=10)
    # 중요: create_combined_anomalies에서 이상치는 결측이 아닌곳에만 생성됨.
    # 따라서 sample_is_outlier_sampled는 순수 이상치를 가리킴.
    if np.any(sample_is_outlier_sampled):
        plt.scatter(synth_pca[sample_is_outlier_sampled, 0], synth_pca[sample_is_outlier_sampled, 1], alpha=0.7, label=f"이상치 ({data_label})", color="purple", marker="x", s=30)
    if np.any(sample_is_missing_sampled):
        plt.scatter(synth_pca[sample_is_missing_sampled, 0], synth_pca[sample_is_missing_sampled, 1], alpha=0.6, label=f"결측 ({data_label})", color="orange", marker="o", s=30, facecolors="none")

    plt.title(f"PCA of {data_label}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.suptitle(f"PCA 비교 (샘플 수: {sample_size})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"{prefix}pca_comparison_combined.png")
    plt.close()
    print(f"PCA 비교 (결합) 그래프 저장 완료: {output_dir / f'{prefix}pca_comparison_combined.png'}")


def plot_tsne_comparison_combined(original_data, synthetic_data, is_outlier_mask, is_missing_mask, data_label, sample_size=1000, perplexity=30, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""):
    """t-SNE를 이용한 2D 시각화 비교 (정상, 이상치, 결측치 구분)"""
    num_time_orig, num_nodes_orig, _, _ = original_data.shape
    num_time_synth, num_nodes_synth, _, _ = synthetic_data.shape

    orig_reshaped = original_data.reshape(num_time_orig * num_nodes_orig, -1)
    synth_reshaped_no_nan = np.nan_to_num(synthetic_data.reshape(num_time_synth * num_nodes_synth, -1), nan=0.0)

    sample_is_outlier = np.any(is_outlier_mask.reshape(num_time_synth * num_nodes_synth, -1), axis=1)
    sample_is_missing = np.any(is_missing_mask.reshape(num_time_synth * num_nodes_synth, -1), axis=1)

    indices_orig = np.random.choice(orig_reshaped.shape[0], min(orig_reshaped.shape[0], sample_size), replace=False)
    orig_sampled = orig_reshaped[indices_orig]

    indices_synth = np.random.choice(synth_reshaped_no_nan.shape[0], min(synth_reshaped_no_nan.shape[0], sample_size), replace=False)
    synth_sampled = synth_reshaped_no_nan[indices_synth]
    sample_is_outlier_sampled = sample_is_outlier[indices_synth]
    sample_is_missing_sampled = sample_is_missing[indices_synth]

    # 원본과 합성을 합쳐서 t-SNE 실행 (상대적 위치 파악)
    combined_data_sampled = np.vstack((orig_sampled, synth_sampled))

    print(f"t-SNE 계산 중... (샘플: {combined_data_sampled.shape[0]}, 시간 소요)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300, init="pca", learning_rate="auto")
    combined_tsne = tsne.fit_transform(combined_data_sampled)

    orig_tsne = combined_tsne[: len(orig_sampled)]
    synth_tsne = combined_tsne[len(orig_sampled) :]

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_tsne[:, 0], orig_tsne[:, 1], alpha=0.5, label="원본 데이터", s=10)
    plt.title("t-SNE of Original Data")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    normal_flags_sampled = ~sample_is_outlier_sampled & ~sample_is_missing_sampled

    plt.scatter(synth_tsne[normal_flags_sampled, 0], synth_tsne[normal_flags_sampled, 1], alpha=0.4, label=f"정상 ({data_label})", color="green", s=10)
    if np.any(sample_is_outlier_sampled):
        plt.scatter(synth_tsne[sample_is_outlier_sampled, 0], synth_tsne[sample_is_outlier_sampled, 1], alpha=0.7, label=f"이상치 ({data_label})", color="purple", marker="x", s=30)
    if np.any(sample_is_missing_sampled):
        plt.scatter(synth_tsne[sample_is_missing_sampled, 0], synth_tsne[sample_is_missing_sampled, 1], alpha=0.6, label=f"결측 ({data_label})", color="orange", marker="o", s=30, facecolors="none")

    plt.title(f"t-SNE of {data_label}")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.suptitle(f"t-SNE 비교 (샘플 수: {sample_size}, Perplexity: {perplexity})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"{prefix}tsne_comparison_combined.png")
    plt.close()
    print(f"t-SNE 비교 (결합) 그래프 저장 완료: {output_dir / f'{prefix}tsne_comparison_combined.png'}")


def plot_spatial_distribution_at_time(
    original_data, synthetic_data, data_label, is_outlier_mask, is_missing_mask, timestamp_idx=100, feature_index=0, channel_index=0, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=""
):
    """특정 시점에서의 센서 값 공간 분포 비교 (이상치/결측치 강조)"""
    orig_spatial = original_data[timestamp_idx, :, feature_index, channel_index]
    synth_spatial_raw = synthetic_data[timestamp_idx, :, feature_index, channel_index]  # NaN 포함

    outlier_flags_spatial = is_outlier_mask[timestamp_idx, :, feature_index, channel_index]
    missing_flags_spatial = is_missing_mask[timestamp_idx, :, feature_index, channel_index]

    num_sensors = original_data.shape[1]
    sensor_ids = np.arange(num_sensors)

    min_val = np.nanmin(np.concatenate([orig_spatial, synth_spatial_raw]))
    max_val = np.nanmax(np.concatenate([orig_spatial, synth_spatial_raw]))
    if np.isnan(min_val) or np.isnan(max_val):  # 모든 값이 NaN인 경우 방지
        min_val, max_val = 0, 1

    plt.figure(figsize=(18, 10))

    # 원본 데이터 공간 분포
    plt.subplot(2, 1, 1)
    plt.bar(sensor_ids, orig_spatial, color="skyblue", label="원본 값")
    plt.title(f"원본 데이터 공간 분포 (시간: {timestamp_idx}, 피처: {feature_index}, 채널: {channel_index})")
    plt.xlabel("센서 ID")
    plt.ylabel("값")
    plt.ylim(min_val, max_val)
    plt.xticks(ticks=sensor_ids[::5], labels=sensor_ids[::5], rotation=45, ha="right", fontsize=8)  # 모든 센서 ID 표시하면 너무 많음
    plt.grid(True, axis="y", alpha=0.5)
    plt.legend()

    # 합성 데이터 공간 분포
    plt.subplot(2, 1, 2)
    # 정상 값은 막대로
    normal_sensor_ids = sensor_ids[~outlier_flags_spatial & ~missing_flags_spatial]
    if normal_sensor_ids.size > 0:
        plt.bar(normal_sensor_ids, synth_spatial_raw[normal_sensor_ids], color="lightcoral", label=f"정상 값 ({data_label})")

    # 이상치 값은 다른 색/마커로 (실제 값으로)
    outlier_sensor_ids = sensor_ids[outlier_flags_spatial]
    if outlier_sensor_ids.size > 0:
        plt.scatter(outlier_sensor_ids, synth_spatial_raw[outlier_flags_spatial], color="purple", marker="x", s=80, label=f"이상치 ({data_label})", zorder=5)

    # 결측치 값은 다른 색/마커로 (원본 값 위치에 표시)
    missing_sensor_ids = sensor_ids[missing_flags_spatial]
    if missing_sensor_ids.size > 0:
        plt.scatter(missing_sensor_ids, orig_spatial[missing_flags_spatial], edgecolor="orange", facecolor="none", marker="o", s=80, label=f"결측치 ({data_label})", zorder=5)

    plt.title(f"{data_label} 공간 분포 (시간: {timestamp_idx}, 피처: {feature_index}, 채널: {channel_index})")
    plt.xlabel("센서 ID")
    plt.ylabel("값")
    plt.ylim(min_val, max_val)
    plt.xticks(ticks=sensor_ids[::5], labels=sensor_ids[::5], rotation=45, ha="right", fontsize=8)
    plt.grid(True, axis="y", alpha=0.5)
    plt.legend()

    plt.suptitle(f"시간 {timestamp_idx}에서의 공간 분포 비교", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / f"spatial_distribution_time{timestamp_idx}_combined.png")
    plt.close()
    print(f"공간 분포 (결합) 비교 그래프 저장 완료: {output_dir / f'spatial_distribution_time{timestamp_idx}_combined.png'}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- 사용자 설정 ---
    # 1. 원본 데이터 파일 경로
    original_data_file = STGAN_DATA_PATH / "data.npy"

    # 2. 분석할 합성 데이터 폴더 경로 (create_combined_anomalies.py로 생성된 폴더)
    #    !!! 아래 경로를 실제 생성된 폴더명으로 수정해야 합니다 !!!
    # 예시: python ./datasets/data_pipeline/create_combined_anomalies.py --scenario point --output_dir ./datasets/bay/combined_test
    # 위 명령 실행 후 ./datasets/bay/combined_test/combined_point_dev0.20-0.50_p0.0100_fault0.0015_noise0.0500_interval0.00-1.00
    # 와 같은 폴더가 생성됩니다.
    SYNTHETIC_COMBINED_FOLDER_PATH = Path("./datasets/bay/combined/combined_point_dev0.20-0.50_p0.0500_fault0.0015_noise0.0500_interval0.00-1.00")  # !!! 실제 경로로 수정 !!!
    SYNTHETIC_DATA_NAME = "합성 데이터 (이상치+결측치)"  # 그래프 제목 등에 사용될 이름

    # --- 데이터 로드 ---
    try:
        if not SYNTHETIC_COMBINED_FOLDER_PATH.exists():
            raise FileNotFoundError(f"지정된 합성 데이터 폴더를 찾을 수 없습니다: {SYNTHETIC_COMBINED_FOLDER_PATH}\n" f"먼저 `create_combined_anomalies.py`를 실행하여 해당 폴더를 생성해주세요.")

        original_pems_data = load_original_pems_data(original_data_file)
        synthetic_pems_data, is_outlier_mask, is_missing_mask = load_combined_anomalies_pems_data(SYNTHETIC_COMBINED_FOLDER_PATH)

        # 시각화 결과 저장용 접두사 (폴더명 기반으로 자동 생성)
        viz_prefix = SYNTHETIC_COMBINED_FOLDER_PATH.name + "_"

        # --- 기본 통계 출력 ---
        print_basic_stats_combined(original_pems_data, name="원본 데이터")
        print_basic_stats_combined(synthetic_pems_data, name=SYNTHETIC_DATA_NAME, outlier_mask=is_outlier_mask, missing_mask=is_missing_mask)

        # --- 시각화 실행 ---
        # PEMS-BAY 데이터는 보통 (시간, 노드, [속도, 통행량 등 특징], 채널수) 형태.
        # 여기서는 첫번째 특징(feature_index=0), 첫번째 채널(channel_index=0)을 주로 사용.
        # 데이터셋에 따라 feature/channel 인덱스 조정 필요.

        # 1. 값 분포 비교
        plot_value_distribution(original_pems_data, synthetic_pems_data, data_label=SYNTHETIC_DATA_NAME, feature_index=0, channel_index=0, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 2. 특정 센서 시계열 비교 (이상치/결측치 구분)
        plot_time_series_comparison_combined(
            original_pems_data,
            synthetic_pems_data,
            is_outlier_mask,
            is_missing_mask,
            data_label=SYNTHETIC_DATA_NAME,
            sensor_index=10,
            feature_index=0,
            channel_index=0,
            time_points=700,
            output_dir=OUTPUT_VIZ_DIR_COMBINED,
            prefix=viz_prefix,
        )

        # 3. 이상치 발생 분포 히트맵
        plot_anomaly_mask_heatmap(is_outlier_mask, anomaly_type_label="이상치(Outlier)", output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 4. 결측치 발생 분포 히트맵
        plot_anomaly_mask_heatmap(is_missing_mask, anomaly_type_label="결측치(Missing)", output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix)

        # 5. PCA 비교 (정상/이상치/결측치 구분)
        plot_pca_comparison_combined(
            original_pems_data, synthetic_pems_data, is_outlier_mask, is_missing_mask, data_label=SYNTHETIC_DATA_NAME, sample_size=2000, output_dir=OUTPUT_VIZ_DIR_COMBINED, prefix=viz_prefix
        )

        # 6. t-SNE 비교 (정상/이상치/결측치 구분, 시간 소요)
        plot_tsne_comparison_combined(
            original_pems_data,
            synthetic_pems_data,
            is_outlier_mask,
            is_missing_mask,
            data_label=SYNTHETIC_DATA_NAME,
            sample_size=1000,
            perplexity=30,  # 샘플 수 줄여서 테스트
            output_dir=OUTPUT_VIZ_DIR_COMBINED,
            prefix=viz_prefix,
        )

        # 7. 특정 시점 공간 분포 비교 (이상치/결측치 구분)
        plot_spatial_distribution_at_time(
            original_pems_data,
            synthetic_pems_data,
            data_label=SYNTHETIC_DATA_NAME,
            is_outlier_mask=is_outlier_mask,
            is_missing_mask=is_missing_mask,
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
        print(f"  - 합성 데이터 폴더 예상 경로: {SYNTHETIC_COMBINED_FOLDER_PATH}")
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
