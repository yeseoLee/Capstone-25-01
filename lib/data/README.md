# lib/data: 데이터 처리 모듈

이 디렉토리는 GRIN 모델에서 사용하는 시계열 데이터 처리 파이프라인을 구현한 클래스와 유틸리티를 포함하고 있습니다.

## 구조

```
lib/data/
├── __init__.py                   # 기본 import 정의
├── imputation_dataset.py         # 결측치 처리 데이터셋
├── spatiotemporal_dataset.py     # 공간-시간 데이터셋
├── temporal_dataset.py           # 시계열 데이터셋 기본 클래스
├── preprocessing/                # 데이터 전처리 도구
│   ├── __init__.py
│   └── scalers.py                # 데이터 스케일링
└── datamodule/                   # PyTorch Lightning 데이터 모듈
    ├── __init__.py
    └── spatiotemporal.py         # 공간-시간 데이터 모듈
```

## 주요 클래스

### 기본 데이터셋

- **TemporalDataset** (`temporal_dataset.py`): 시계열 데이터에 대한 기본 데이터셋 클래스

  - 시간 윈도우 슬라이딩 기반 데이터 처리
  - 외부 데이터(exogenous variables) 통합 기능
  - 트렌드 제거 및 스케일링을 포함한 전처리 기능
- **SpatioTemporalDataset** (`spatiotemporal_dataset.py`): 공간적 차원을 포함한 시계열 데이터셋

  - 다중 노드(센서) 데이터 처리
  - 데이터 형태: [시간 단계, 노드, 특성]

### 결측치 처리 데이터셋

- **ImputationDataset** (`imputation_dataset.py`): 결측치 처리를 위한 데이터셋

  - 마스크 처리 기능 (결측값 위치 표시)
  - 평가용 마스크 지원
- **GraphImputationDataset** (`imputation_dataset.py`): 그래프 기반 결측치 처리 데이터셋

  - ImputationDataset과 SpatioTemporalDataset 다중 상속
  - GRIN 모델의 핵심 데이터셋 클래스

### 전처리 도구

- **AbstractScaler** (`preprocessing/scalers.py`): 스케일러의 추상 기본 클래스
- **StandardScaler** (`preprocessing/scalers.py`): 표준화 스케일러 (평균=0, 표준편차=1)
- **MinMaxScaler** (`preprocessing/scalers.py`): 최소-최대 정규화 스케일러 (범위: 0~1)

### 데이터 모듈

- **SpatioTemporalDataModule** (`datamodule/spatiotemporal.py`): 공간-시간 데이터 처리 모듈
  - PyTorch Lightning LightningDataModule 상속
  - 데이터셋 분할 관리 (학습/검증/테스트)
  - 데이터 로딩 및 배치 생성
  - 전처리 파이프라인 통합

## 데이터 처리 파이프라인

1. **데이터 로딩**: 원시 데이터 로드
2. **데이터셋 생성**: 적절한 데이터셋 클래스로 변환
3. **마스킹 처리**: 결측값 마스킹
4. **전처리**: 트렌드 제거 및 스케일링
5. **데이터 분할**: 학습/검증/테스트 세트 분할
6. **배치 생성**: 모델 학습용 배치 생성

## 사용 예시

```python
# 데이터셋 생성
dataset = GraphImputationDataset(
    data,             # 원시 데이터: shape [시간, 노드, 특성]
    mask=mask,        # 마스크: 유효한 데이터 = 1, 결측값 = 0
    window=24,        # 입력 윈도우 크기
    horizon=24,       # 예측 수평선 크기
    stride=1          # 슬라이딩 윈도우 스트라이드
)

# 데이터모듈 생성
datamodule = SpatioTemporalDataModule(
    dataset,
    train_idxs=train_indices,
    val_idxs=val_indices,
    test_idxs=test_indices,
    batch_size=32,
    scale=True,
    scaling_type='std'
)

# 데이터모듈 설정
datamodule.setup()

# 데이터로더 가져오기
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

이 데이터 처리 모듈은 GRIN 모델에서 사용되는 교통 네트워크 데이터셋(METR-LA, PEMS-BAY)과 같은 시공간 데이터의 결측치 처리를 위해 설계되었습니다.
