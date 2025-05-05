# 이상치 및 결측치 공존 상황에서의 교통 데이터 보정

이 프로젝트는 교통 데이터에서 이상치와 결측치가 동시에 존재하는 상황에서의 보정 파이프라인을 연구합니다.

## 프로젝트 개요

- **목적**: 이상치와 결측치가 공존하는 교통 데이터에서 효과적인 보정 파이프라인 연구
- **데이터**: PEMS-BAY, METR-LA 교통 데이터셋
- **접근법**: 그래프 신경망(GNN) 기반의 모델을 이용한 다양한 보간 파이프라인 비교
  - 이상치 보간 모델: GCNODE (Graph Convolutional Network for Outlier Detection and Estimation)
  - 결측치 보간 모델: GCNMI (Graph Convolutional Network for Missing Value Imputation)
  - 파이프라인 비교: 다양한 보간 전략 비교

## 주요 파이프라인

1. **결측치 → 이상치 보간 파이프라인**: 결측치를 먼저 보간한 후 이상치를 보간 (결측치 보간 시 이상치를 정상값으로 간주)
2. **임시 대체 → 이상치 → 결측치 보간 파이프라인**: 결측치에 임시 값(0, 평균, 이웃 값)을 채운 후 이상치 보간, 이후 결측치 재보간
3. **번갈아 보간 파이프라인**: 결측치와 이상치 보간을 번갈아 가며 반복적으로 수행

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
