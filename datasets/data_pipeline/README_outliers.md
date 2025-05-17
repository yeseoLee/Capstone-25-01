# STGAN 이상치 생성 스크립트

이 스크립트는 STGAN 모델에서 사용하는 PemsBay 데이터셋에 다양한 유형의 이상치(outlier)를 생성하고 저장합니다.

## 개요

교통 데이터에는 다양한 원인으로 이상치가 발생할 수 있습니다. 이 스크립트는 세 가지 유형의 이상치를 생성할 수 있습니다:

1. **점 이상치(Point Outlier)**: 임의의 데이터 포인트에서 정상 범위를 벗어난 값을 갖는 이상치
2. **블록 이상치(Block Outlier)**: 연속된 시간대에 걸쳐 이상치가 발생하는 경우
3. **맥락 이상치(Contextual Outlier)**: 특정 맥락에서 비정상적인 패턴을 보이는 이상치
   - 출퇴근 시간대에 한산한 새벽 시간대 패턴 복사
   - 평일 낮 시간대에 주말 낮 시간대 패턴 복사

## 사용법

### 기본 사용법:

다음 명령어로 이상치를 생성할 수 있습니다:

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario point
```

기본 설정으로는 PemsBay 데이터셋에 점 이상치를 생성합니다.

### 매개변수 설정:

```bash
python ./datasets/data_pipeline/create_outliers.py --dataset_name bay --output_dir ./datasets/bay/outliers --scenario block --min_deviation 0.3 --max_deviation 0.6 --min_duration 10 --max_duration 30
```

### 매개변수 설명:

#### 공통 매개변수:

- `--dataset_name`: 데이터셋 이름 (현재는 'bay'만 지원)
- `--output_dir`: 출력 디렉토리 경로
- `--scenario`: 이상치 생성 시나리오 ('point', 'block', 'contextual')
- `--seed`: 랜덤 시드

#### 점 이상치 매개변수:

- `--min_deviation`: 최소 편차 (정규화된 값)
- `--max_deviation`: 최대 편차 (정규화된 값)
- `--p_outlier`: 이상치 생성 확률

#### 블록 이상치 매개변수:

- `--min_deviation`: 최소 편차 (정규화된 값)
- `--max_deviation`: 최대 편차 (정규화된 값)
- `--min_duration`: 최소 지속 시간
- `--max_duration`: 최대 지속 시간
- `--p_outlier`: 이상치 생성 확률

#### 맥락 이상치 매개변수:

- `--replace_ratio`: 대체 비율 (맥락 이상치를 생성할 데이터의 비율)

## 출력 결과

스크립트 실행 결과로 다음과 같은 폴더 구조가 생성됩니다:

```
STGAN/bay/outliers/
├── point_dev0.20-0.50_p0.0100/           # 점 이상치 폴더
│   ├── data.npy                          # 이상치가 포함된 데이터
│   ├── outlier_mask.npy                  # 이상치 마스크 (True: 정상, False: 이상치)
│   ├── time_features.txt                 # 시간 특성 (원본에서 복사)
│   ├── node_subgraph.npy                 # 노드 서브그래프 (원본에서 복사)
│   ├── node_adjacent.txt                 # 노드 인접 정보 (원본에서 복사)
│   └── node_dist.txt                     # 노드 거리 정보 (원본에서 복사)
├── block_dev0.20-0.50_dur5-20_p0.0050/   # 블록 이상치 폴더
│   └── ...
└── contextual_ratio0.05/                 # 맥락 이상치 폴더
    └── ...
```

## 이상치 유형별 설명

### 1. 점 이상치 (Point Outlier)

개별 데이터 포인트에 무작위로 편차를 추가합니다. 센서 오작동, 통신 오류 등으로 인한 일시적인 이상치를 시뮬레이션합니다.

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario point --min_deviation 0.2 --max_deviation 0.5 --p_outlier 0.01
```

### 2. 블록 이상치 (Block Outlier)

연속된 시간 구간에 걸쳐 편차를 추가합니다. 센서 고장, 교통 사고, 특별 행사 등으로 인한 지속적인 이상치를 시뮬레이션합니다.

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario block --min_duration 10 --max_duration 30 --p_outlier 0.005
```

### 3. 맥락 이상치 (Contextual Outlier)

특정 맥락(시간대, 요일 등)에서 비정상적인 패턴을 생성합니다. 비정상적인 교통 패턴 변화를 시뮬레이션합니다.

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario contextual --replace_ratio 0.1
```

맥락 이상치는 다음과 같은 방식으로 생성됩니다:

- 출퇴근 시간대(8-9시, 17-18시)와 새벽 시간대(2-4시) 데이터 교환
- 평일 낮 시간대(10-16시)와 주말 낮 시간대(10-16시) 데이터 교환

## 시간 특성 데이터 형식

`time_features.txt` 파일은 각 타임스탬프의 시간 정보를 담고 있는 원-핫 인코딩 형태의 데이터입니다:

- 첫 7개 열: 요일 정보 (월~일, 원-핫 인코딩)
- 다음 24개 열: 시간대 정보 (0~23시, 원-핫 인코딩)

이 정보를 바탕으로 맥락 이상치를 생성할 때 적절한 시간대와 요일을 선택합니다.

## 예제

### 1. 점 이상치 생성 (기본값)

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario point
```

### 2. 블록 이상치 생성 (지속 시간 설정)

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario block --min_duration 15 --max_duration 45
```

### 3. 맥락 이상치 생성 (대체 비율 증가)

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario contextual --replace_ratio 0.2
```

## 시나리오별 실행 방법

### 점 이상치

점 이상치는 특정 시간대에 정상 값의 범위를 벗어난 갑작스러운 변동을 의미합니다.

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario point --min_deviation 0.2 --max_deviation 0.5 --p_outlier 0.01
```

### 블록 이상치

블록 이상치는 일정 기간 동안 정상 값의 범위를 벗어난 지속적인 변동을 의미합니다.

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario block --min_duration 10 --max_duration 30 --p_outlier 0.005
```

### 맥락적 이상치

맥락적 이상치는 정상 데이터 범위 내에 있지만 다른 시간대의 패턴이 현재 맥락과 맞지 않는 경우를 의미합니다.

```bash
python ./datasets/data_pipeline/create_outliers.py --scenario contextual --replace_ratio 0.1
```

## 다양한 예시

다음은 이상치를 생성하기 위한 몇 가지 예시 명령어입니다:

```bash
# 점 이상치 (기본 설정)
python ./datasets/data_pipeline/create_outliers.py --scenario point

# 블록 이상치 (지속 시간 변경)
python ./datasets/data_pipeline/create_outliers.py --scenario block --min_duration 15 --max_duration 45

# 맥락적 이상치 (비율 변경)
python ./datasets/data_pipeline/create_outliers.py --scenario contextual --replace_ratio 0.2
```
