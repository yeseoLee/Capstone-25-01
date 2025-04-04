# 이상치 및 결측치 공존 상황에서의 교통 데이터 보정

이 프로젝트는 교통 데이터에서 이상치와 결측치가 동시에 존재하는 상황에서의 보정 방법론을 연구합니다.

## 프로젝트 개요

- **목적**: 이상치와 결측치가 공존하는 교통 데이터에서 효과적인 보정 방법 연구
- **데이터**: PEMS-BAY, METR-LA 교통 데이터셋
- **접근법**: 그래프 신경망(GNN) 기반의 모델을 이용한 보간 방법 비교
  - 이상치 보간 모델: GCNODE, GATODE, GraphSAGEODE
  - 결측치 보간 모델: GCNMI, BIGSAGE, GATEDGAT
  - 방법론 비교: 이상치→결측치, 결측치→이상치 순차 보간

## 프로젝트 구조

- `data/`: 데이터셋 관련 스크립트
  - `generate_data.py`: 결측치 및 이상치가 포함된 데이터 생성
- `models/`: 다양한 보정 모델 구현
  - `base_model.py`: 모델 기본 클래스
  - `outlier_models.py`: 이상치 보간 모델
  - `missing_models.py`: 결측치 보간 모델
- `utils/`: 유틸리티 함수
  - `data_utils.py`: 데이터 전처리 함수
  - `metrics.py`: 평가 지표 계산 함수
  - `visualization.py`: 시각화 함수
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

- 모델 가중치: `experiments/models/`
- 평가 결과: `experiments/results/`

## 요구사항

- Python 3.8 이상
- PyTorch 1.10 이상
- PyTorch Geometric 2.0 이상
