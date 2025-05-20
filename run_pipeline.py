"""
결측치와 이상치를 보간하는 파이프라인

사용법:
    python run_pipeline.py --output_dir ./datasets/bay/pipeline_results --outlier_scenario point --missing_type block --node_ratio 0.1

파라미터 설명:
    --dataset_name: 데이터셋 이름 (현재는 'bay'만 지원)
    --output_dir: 출력 디렉토리 경로
    --outlier_scenario: 이상치 생성 시나리오 ('point', 'block', 'contextual')
    --missing_type: 결측치 유형 ('block', 'point')
    --node_ratio: 전체 노드 중 사용할 비율 (0.0-1.0)
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def run_command(command, description):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*50}")
    print(f"{description} 실행 중...")
    print(f"명령어: {command}")
    print(f"{'='*50}\n")

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"에러 출력: {e.stderr}")
        return False


def create_dataset_with_anomalies(dataset_name, output_dir, outlier_scenario="point", missing_type="block"):
    """이상치와 결측치가 포함된 데이터셋을 생성합니다."""
    # 1. 이상치 생성
    outlier_dir = os.path.join(output_dir, "anomaly_data")
    outlier_cmd = f"python datasets/data_pipeline/create_outliers.py --dataset_name {dataset_name} --output_dir {outlier_dir} --scenario {outlier_scenario}"

    if not run_command(outlier_cmd, "이상치 생성"):
        return None

    # 이상치 생성 스크립트의 출력에서 데이터셋 경로 파싱
    try:
        result = subprocess.run(outlier_cmd, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout

        # 출력에서 데이터셋 경로 찾기
        for line in output.split("\n"):
            if "이상치가 포함된 데이터가" in line:
                dataset_path = line.split("이상치가 포함된 데이터가")[1].split("에 저장되었습니다")[0].strip()
                break
        else:
            print("이상치 데이터셋 경로를 찾을 수 없습니다.")
            return None

        # 2. 생성된 이상치 데이터셋에 결측치 추가
        missing_cmd = f"python datasets/data_pipeline/create_missing_data.py --dataset_name {dataset_name} --output_dir {dataset_path} --mask_type {missing_type}"

        if not run_command(missing_cmd, "결측치 생성"):
            return None

        return dataset_path

    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"에러 출력: {e.stderr}")
        return None
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        return None


def select_nodes(dataset_name, node_ratio):
    """사용할 노드 리스트를 생성합니다."""
    # 데이터셋의 전체 노드 수 확인
    data_path = f"datasets/{dataset_name}/data/data.npy"
    if not os.path.exists(data_path):
        print(f"데이터셋을 찾을 수 없습니다: {data_path}")
        return None

    data = np.load(data_path)
    total_nodes = data.shape[1]  # (time, nodes, features, channels)

    # 노드 선택
    num_nodes = int(total_nodes * node_ratio)
    selected_nodes = np.random.choice(total_nodes, num_nodes, replace=False)
    selected_nodes.sort()  # 정렬된 노드 리스트 반환

    return ",".join(map(str, selected_nodes))


def train_models(dataset_name, output_dir, node_list):
    """STGAN과 SPIN 모델을 학습합니다."""
    # STGAN 학습
    stgan_cmd = f"python STGAN_custom/main.py --dataset {dataset_name} --root_path . --use-node-subset --node-list {node_list}"

    if not run_command(stgan_cmd, "STGAN 모델 학습"):
        return False

    # SPIN 학습
    spin_cmd = f"python SPIN_custom/train.py --dataset {dataset_name} --root_path . --use-node-subset --node-list {node_list}"

    if not run_command(spin_cmd, "SPIN 모델 학습"):
        return False

    return True


def detect_and_replace_anomalies(dataset_name, output_dir, node_list):
    """STGAN을 사용하여 이상치를 탐지하고 0으로 대체합니다."""
    # STGAN 테스트 실행
    stgan_test_cmd = f"python STGAN_custom/outlier_detection.py --dataset {dataset_name} --root_path . --use-node-subset --node-list {node_list}"

    if not run_command(stgan_test_cmd, "STGAN 이상치 탐지"):
        return None

    # 결과 파일 경로
    result_path = os.path.join(output_dir, "stgan_results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path


def evaluate_with_spin(dataset_name, output_dir, node_list):
    """SPIN 모델을 사용하여 보정하고 MAE, MSE를 계산합니다."""
    # SPIN 테스트 실행
    model_path = os.path.join(output_dir, "checkpoints/model.pth")
    data_path = os.path.join(output_dir, "stgan_results/data.npy")

    spin_test_cmd = f"python SPIN_custom/test.py --dataset {dataset_name} --root_path . --model_path {model_path} --data_path {data_path} --use-node-subset --node-list {node_list}"

    if not run_command(spin_test_cmd, "SPIN 모델 평가"):
        return None

    # 결과 파일 경로
    result_path = os.path.join(output_dir, "spin_results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path


def main(args):
    """메인 파이프라인 실행"""
    try:
        # 출력 디렉토리 생성
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 노드 리스트 생성
        node_list = select_nodes(args.dataset_name, args.node_ratio)
        if not node_list:
            print("노드 선택 실패")
            return 1

        # 1. 데이터셋 준비 (이상치 생성 후 결측치 추가)
        print("\n1. 이상치와 결측치가 포함된 데이터셋 생성 중...")
        dataset_path = create_dataset_with_anomalies(args.dataset_name, args.output_dir, args.outlier_scenario, args.missing_type)

        if not dataset_path:
            print("데이터셋 생성 실패")
            return 1

        # 2. 모델 학습
        print("\n2. STGAN과 SPIN 모델 학습 중...")
        if not train_models(args.dataset_name, args.output_dir, node_list):
            print("모델 학습 실패")
            return 1

        # 3. 이상치 탐지 및 대체
        print("\n3. STGAN을 사용한 이상치 탐지 및 대체 중...")
        stgan_results = detect_and_replace_anomalies(args.dataset_name, args.output_dir, node_list)
        if not stgan_results:
            print("이상치 탐지 실패")
            return 1

        # 4. SPIN으로 보정 및 평가
        print("\n4. SPIN 모델을 사용한 보정 및 평가 중...")
        spin_results = evaluate_with_spin(args.dataset_name, args.output_dir, node_list)
        if not spin_results:
            print("SPIN 평가 실패")
            return 1

        print("\n파이프라인 실행 완료!")
        print(f"결과가 {args.output_dir}에 저장되었습니다.")

        return 0

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="결측치와 이상치를 보간하는 파이프라인")
    parser.add_argument("--dataset_name", type=str, default="bay", help="데이터셋 이름 (현재는 bay만 지원)")
    parser.add_argument("--output_dir", type=str, default="./datasets/bay/pipeline_results", help="출력 디렉토리 경로")
    parser.add_argument("--outlier_scenario", type=str, default="point", choices=["point", "block", "contextual"], help="이상치 생성 시나리오")
    parser.add_argument("--missing_type", type=str, default="block", choices=["block", "point"], help="결측치 유형")
    parser.add_argument("--node_ratio", type=float, default=0.2, help="전체 노드 중 사용할 비율 (0.0-1.0)")

    args = parser.parse_args()
    sys.exit(main(args))
