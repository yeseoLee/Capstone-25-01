import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_traffic_graph(adj_matrix, node_values=None, node_size=100, figsize=(12, 10)):
    """
    교통 네트워크 그래프 시각화
    Args:
        adj_matrix: 인접 행렬
        node_values: 노드 값 (색상 표현용)
        node_size: 노드 크기
        figsize: 그림 크기
    """
    # 그래프 생성
    G = nx.from_numpy_matrix(adj_matrix)

    plt.figure(figsize=figsize)

    # 노드 포지션 계산 (spring layout)
    pos = nx.spring_layout(G, seed=42)

    # 노드 색상 설정
    if node_values is not None:
        # 현재 시점의 노드 값 사용 (평균값)
        if len(node_values.shape) > 1:
            node_values_mean = np.nanmean(node_values, axis=1)
        else:
            node_values_mean = node_values

        # 노드 색상 매핑
        vmin = np.nanmin(node_values_mean)
        vmax = np.nanmax(node_values_mean)

        # 노드 그리기
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=node_values_mean,
            cmap=plt.cm.viridis,
            vmin=vmin,
            vmax=vmax,
        )

        # 컬러바 추가
        plt.colorbar(nodes)
    else:
        # 노드 값이 없을 경우 기본 색상으로 그리기
        nx.draw_networkx_nodes(G, pos, node_size=node_size)

    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    # 노드 레이블 (노드 수가 적을 경우에만 표시)
    if adj_matrix.shape[0] <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("교통 네트워크 그래프")
    plt.axis("off")

    return plt.gcf()


def plot_time_series(
    original_data,
    corrupted_data,
    imputed_data,
    node_idx,
    time_range=None,
    figsize=(14, 6),
):
    """
    시계열 데이터 시각화 (원본, 손상된 데이터, 보간된 데이터)
    Args:
        original_data: 원본 데이터
        corrupted_data: 손상된 데이터 (결측치/이상치 포함)
        imputed_data: 보간된 데이터
        node_idx: 시각화할 노드 인덱스
        time_range: 시각화할 시간 범위 (None일 경우 전체)
        figsize: 그림 크기
    """
    if time_range is None:
        time_range = slice(0, original_data.shape[1])

    plt.figure(figsize=figsize)

    time_idx = np.arange(original_data.shape[1])[time_range]

    # 원본 데이터
    plt.plot(
        time_idx,
        original_data[node_idx, time_range],
        "b-",
        label="원본 데이터",
        alpha=0.7,
    )

    # 손상된 데이터 (NaN이 아닌 부분만)
    mask = ~np.isnan(corrupted_data[node_idx, time_range])
    plt.scatter(
        time_idx[mask],
        corrupted_data[node_idx, time_range][mask],
        c="r",
        marker="x",
        label="손상된 데이터",
        alpha=0.7,
    )

    # 보간된 데이터
    if imputed_data is not None:
        plt.plot(
            time_idx,
            imputed_data[node_idx, time_range],
            "g--",
            label="보간된 데이터",
            alpha=0.7,
        )

    plt.xlabel("시간")
    plt.ylabel("값")
    plt.title(f"노드 {node_idx}의 시계열 데이터")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def plot_imputation_comparison(
    original_data,
    corrupted_data,
    imputed_data_dict,
    node_idx,
    time_range=None,
    figsize=(16, 8),
):
    """
    여러 모델의 보간 결과 비교 시각화
    Args:
        original_data: 원본 데이터
        corrupted_data: 손상된 데이터 (결측치/이상치 포함)
        imputed_data_dict: 각 모델별 보간 결과 (모델명: 보간 데이터)
        node_idx: 시각화할 노드 인덱스
        time_range: 시각화할 시간 범위 (None일 경우 전체)
        figsize: 그림 크기
    """
    if time_range is None:
        time_range = slice(0, original_data.shape[1])

    plt.figure(figsize=figsize)

    time_idx = np.arange(original_data.shape[1])[time_range]

    # 원본 데이터
    plt.plot(
        time_idx,
        original_data[node_idx, time_range],
        "k-",
        label="원본 데이터",
        linewidth=2,
        alpha=0.7,
    )

    # 손상된 데이터 (NaN이 아닌 부분만)
    mask = ~np.isnan(corrupted_data[node_idx, time_range])
    plt.scatter(
        time_idx[mask],
        corrupted_data[node_idx, time_range][mask],
        c="r",
        marker="x",
        label="손상된 데이터",
        alpha=0.7,
    )

    # 각 모델별 보간 결과
    colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
    linestyles = ["--", "-.", ":", "--", "-.", ":"]

    for i, (model_name, imputed_data) in enumerate(imputed_data_dict.items()):
        color_idx = i % len(colors)
        style_idx = i % len(linestyles)
        plt.plot(
            time_idx,
            imputed_data[node_idx, time_range],
            color=colors[color_idx],
            linestyle=linestyles[style_idx],
            label=f"{model_name} 보간",
            alpha=0.7,
        )

    plt.xlabel("시간")
    plt.ylabel("값")
    plt.title(f"노드 {node_idx}의 보간 결과 비교")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def plot_metrics_comparison(metrics_dicts, metric_name="combined_rmse", figsize=(10, 6)):
    """
    여러 모델의 성능 지표 비교 시각화
    Args:
        metrics_dicts: 각 모델별 성능 지표 딕셔너리 (모델명: 지표 딕셔너리)
        metric_name: 비교할 지표 이름
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)

    model_names = list(metrics_dicts.keys())
    metric_values = [metrics_dict[metric_name] for metrics_dict in metrics_dicts.values()]

    # 바 차트로 표현
    bars = plt.bar(model_names, metric_values, alpha=0.7)

    # 바 위에 값 표시
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("모델")
    plt.ylabel(metric_name)
    plt.title(f"모델별 {metric_name} 성능 비교")
    plt.grid(True, alpha=0.3, axis="y")

    # y축 범위 조정 (0부터 시작, 최대값의 10% 여유)
    max_value = max(metric_values)
    plt.ylim(0, max_value * 1.1)

    return plt.gcf()
