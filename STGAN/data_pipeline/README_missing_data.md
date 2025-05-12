# STGAN 결측치 생성 스크립트

이 스크립트는 STGAN 모델에서 사용하는 PemsBay 데이터셋에 결측치를 생성하고 마스킹된 데이터를 저장합니다.

## 개요

- 도로 교통 데이터에는 센서 고장, 통신 오류 등으로 결측치가 발생할 수 있습니다.
- 이 스크립트는 두 가지 유형의 결측치를 생성합니다:
  - **Block 결측치**: 연속된 시간 동안 데이터가 누락되는 경우 (센서 고장 등 시뮬레이션)
  - **Point 결측치**: 랜덤하게 독립적으로 발생하는 결측치 (일시적 오류 시뮬레이션)

## 사용법

### 기본 사용법:

```bash
python ./STGAN/create_missing_data.py
```

기본 설정으로는 PemsBay 데이터셋에 block 타입 결측치를 생성합니다. (결함 확률 0.0015, 잡음 확률 0.05)

### 매개변수 설정:

```bash
python ./STGAN/create_missing_data.py --dataset_name bay --output_dir ./STGAN/bay/masked --p_fault 0.002 --p_noise 0.08 --mask_type block
```

### 매개변수 설명:

- `--dataset_name`: 데이터셋 이름 (현재는 'bay'만 지원)
- `--output_dir`: 출력 디렉토리 경로
- `--p_fault`: 결함 확률 (연속적인 결측치 생성 비율)
- `--p_noise`: 잡음 확률 (독립적인 결측치 생성 비율)
- `--mask_type`: 마스크 유형 ('block' 또는 'point')
  - 'block': 연속적인 결측치 (p_fault, p_noise 모두 사용)
  - 'point': 독립적인 결측치만 생성 (p_noise=0.25 고정, p_fault 무시)
- `--visualize_node`: 시각화할 노드 인덱스 (기본값: 0)
- `--visualize_feature`: 시각화할 특성 인덱스 (0 또는 1, 기본값: 0)
- `--visualize_time`: 시각화할 시간 범위 (기본값: 100)

## 출력 결과

스크립트 실행 결과로 다음과 같은 폴더 구조가 생성됩니다:

```
STGAN/bay/masked/
├── block_fault0.0015_noise0.0500/    # 마스크 유형 및 설정에 따른 폴더명
│   ├── data.npy                      # 결측치가 포함된 데이터 (NaN으로 표시)
│   ├── mask.npy                      # 마스크 (1: 유효한 값, 0: 결측치)
│   ├── missing_data_comparison.png   # 결측치 생성 전후 비교 시각화 이미지
│   ├── time_features.txt             # 시간 특성 (원본에서 복사)
│   ├── node_subgraph.npy             # 노드 서브그래프 (원본에서 복사)
│   ├── node_adjacent.txt             # 노드 인접 정보 (원본에서 복사)
│   └── node_dist.txt                 # 노드 거리 정보 (원본에서 복사)
```

## 결측치 시각화

결측치 데이터를 시각화하기 위한 별도의 Jupyter 노트북이 제공됩니다.

### 시각화 노트북 사용법:

1. 먼저 결측치 생성 스크립트를 실행하여 마스킹된 데이터를 생성합니다:
   ```bash
   python ./STGAN/create_missing_data.py
   ```

2. Jupyter Notebook을 실행하고 `visualize_missing_data.ipynb` 파일을 엽니다:
   ```bash
   jupyter notebook STGAN/visualize_missing_data.ipynb
   ```

3. 노트북 내에서 데이터 경로를 확인하고 필요에 따라 수정합니다:
   ```python
   original_data_path = "./bay/data/data.npy"
   masked_data_path = "./bay/masked/block_fault0.0015_noise0.0500/data.npy"
   mask_path = "./bay/masked/block_fault0.0015_noise0.0500/mask.npy"
   ```

### 시각화 기능:

노트북은 다음과 같은 시각화 기능을 제공합니다:

1. **단일 시계열 시각화**: 특정 노드와 특성에 대한 원본 데이터와 결측 데이터 비교
2. **여러 노드 비교**: 여러 노드의 결측 패턴을 동시에 시각화
3. **결측치 분포 분석**: 노드별/시간별 결측치 비율 분석
4. **결측치 히트맵**: 2D 히트맵으로 결측 패턴 시각화
5. **연속 결측 구간 분석**: 연속적인 결측 구간 길이 분포 확인

## 예제

### 1. Block 결측치 생성 (기본값)

```bash
python ./STGAN/create_missing_data.py
```

### 2. Point 결측치 생성

```bash
python ./STGAN/create_missing_data.py --mask_type point
```

### 3. 사용자 지정 결측 비율 설정

```bash
python ./STGAN/create_missing_data.py --p_fault 0.001 --p_noise 0.1
```
