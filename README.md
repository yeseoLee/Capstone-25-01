# 이상치 및 결측치 공존 상황에서의 교통 데이터 보정

이 프로젝트는 교통 데이터에서 이상치와 결측치가 동시에 존재하는 상황에서의 보정 파이프라인을 연구합니다.

## 프로젝트 개요

- **목적**: 이상치와 결측치가 공존하는 교통 데이터에서 효과적인 보정 파이프라인 연구
- **데이터**: PEMS-BAY, METR-LA 교통 데이터셋
- **접근법**: 그래프 신경망(GNN) 기반의 모델을 이용한 다양한 보간 파이프라인 비교
  - 이상치 탐지 모델: STGAN
  - 결측치 보간 모델: SPIN
  - 파이프라인 비교: 다양한 보간 전략 비교

## 주요 파이프라인

1. **결측치 보간 → 이상치 탐지 파이프라인**: 결측치를 먼저 보간한 후 이상치를 탐지, 이후 이상치를 결측치로 바꿔 보간 (결측치 보간 시 이상치를 정상값으로 간주)
2. **이상치 탐지 -> 결측치 보간 파이프라인**: 이상치를 먼저 탐지하고 결측치로 바꿔 한번에 보간 (이상치 탐지시 결측치를 정상값으로 간주)
3. 결측치 보간 및 이상치 탐지 반복 파이프라인: 탐지 & 보간 N번 반복 수행

 어떤 방식으로 구성하냐에 따라 이상치 탐지 성능이 달라지고, 이상치 보정은 결국 하나의 방식으로 귀결된다.

## 이상치 생성 방법

- 400개의 타임스탬프마다, 검출기당 최소/최대 측정값보다 5~10mph 작거나 큰 이상 현상을 데이터 세트에 추가
  (이상 현상의 지속 시간은 10(50분)에서 50(250분)까지이며, 단계 크기는 10)
- 이상 현상 생성. PEMS08 데이터셋을 사용하여 실제 교통사고의 시공간적 분포를 시뮬레이션
  (감지기 측정값은 300m/1000m/3000m 이내였으며, 사고 지속 시간은 해당 감지기 측정값의 최소값보다 5~10mph 낮은 값으로 설정)
- 순서/시간대 교환: 주말/공휴일과 평일 데이터를 교환, 날씨별 교환, 시간대별 교환, 데이터 순서 교환, 센서 교환 등
- Noise injection approach
- 참고
  - https://www.sciencedirect.com/science/article/pii/S1319157822001665
  - https://arxiv.org/abs/2406.11901

## 프로젝트 구조

- `datasets/`: 데이터셋 관련 스크립트
  - `generate_data.py`: 결측치 및 이상치가 포함된 데이터 생성
- `models/`: 다양한 보정 모델 구현
  - `base_model.py`: 모델 기본 클래스
  - `outlier/`: 이상치 보간 모델
  - `missing/`: 결측치 보간 모델
- `utils/`: 유틸리티 함수
  - `data_utils.py`: 데이터 전처리 함수
  - `metrics.py`: 평가 지표 계산 함수
  - `visualization.py`: 시각화 함수
  - `pipelines.py`: 다양한 보간 파이프라인 구현
- `experiments/`: 실험 스크립트
  - `train.py`: 모델 학습 스크립트
  - `evaluate.py`: 모델 평가 스크립트
- `run.py`: 통합 실행 스크립트

## 설치 방법

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

```bash
# 1. 데이터 생성
python run.py generate

# 2. 모델 학습
python run.py train [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LR]

# 3. 모델 평가
python run.py evaluate [--cpu]

# 또는 모든 과정 한 번에 실행
python run.py all [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LR] [--cpu]
```

## 결과 확인

학습 결과와 모델 평가는 다음 위치에 저장됩니다:

- 모델 가중치: `models/saved/`
- 평가 결과: `results/pipelines/`

## 요구사항

- Python 3.8 이상
- PyTorch 1.10 이상
- PyTorch Geometric 2.0 이상
- NumPy, Matplotlib, scikit-learn
