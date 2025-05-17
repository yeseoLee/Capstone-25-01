# STGAN과 SPIN 모델 통합

이 문서는 STGAN에서 사용하는 데이터셋 형식을 SPIN 모델에서 사용할 수 있도록 변환하는 방법에 대해 설명합니다.

## 개요

STGAN과 SPIN 모델은 각각 다른 데이터셋 형식을 사용하고 있어 통합이 필요했습니다. 이를 위해 세 가지 주요 파일을 개발했습니다:

1. `utils/stgan_dataset.py`: STGAN 데이터셋을 TSL 라이브러리 형식으로 로드하는 클래스
2. `experiments/run_imputation_stgan.py`: STGAN 데이터셋을 사용해 SPIN 모델을 실행하는 스크립트
3. `experiments/run_inference_stgan.py`: 학습된 모델로 결측치를 보정하고 STGAN 형식으로 저장하는 스크립트

## 데이터셋 형식

### STGAN 데이터셋 형식

- `data.npy`: (시간, 노드, 특성, 채널) 형태의 4차원 교통 데이터
- `time_features.txt`: 앞 7개 열은 요일, 뒤 24개 열은 시간대를 원핫인코딩한, 총 31개의 원소로 구성된 시간 특성
- `node_adjacent.txt`: 각 노드에 대해 가장 가까운 노드 8개를 기록한 인접행렬
- `node_dist.txt`: 인접행렬에 표시된 노드들에 대한 거리 행렬

### SPIN 모델 (TSL 라이브러리) 데이터셋 형식

- 3차원 텐서: (시간, 노드, 채널)
- 연결성 행렬: 노드 간의 연결 정보
- 시간 특성: 날짜, 시간 등의 시간 정보

## 데이터 변환 과정

1. `STGANBayDataset` 클래스에서는 다음과 같은 변환을 수행합니다:

   - STGAN의 4차원 데이터 (시간, 노드, 특성, 채널)를 3차원 (시간, 노드, 특성*채널)으로 변환
   - STGAN의 인접 행렬을 TSL에서 사용하는 연결성 행렬 형식으로 변환
   - STGAN의 시간 특성을 TSL에서 사용하는 형식으로 변환
2. `run_imputation_stgan.py` 스크립트에서는:

   - STGAN 데이터셋을 로드
   - TSL 라이브러리의 `add_missing_values`를 사용해 필요한 결측치 패턴 생성
   - SPIN 모델 학습 및 평가 수행
3. `run_inference_stgan.py` 스크립트에서는:

   - 학습된 모델 로드
   - 결측치가 있는 테스트 데이터에 대해 예측 수행
   - 보정된 결과를 원래 STGAN 형식으로 변환하여 저장

## 사용 방법

### 1. 모델 학습

다음 명령어로 STGAN 데이터셋을 사용하여 SPIN 모델을 학습할 수 있습니다:

```bash
cd SPIN
python experiments/run_imputation_stgan.py --root-dir /path/to/datasets/bay --dataset-name bay_block
```

주요 매개변수:

- `--root-dir`: STGAN 데이터셋 경로 (기본값: 현재 작업 디렉토리의 'datasets/bay')
- `--dataset-name`: 데이터셋 이름과 결측치 패턴 (bay_point 또는 bay_block)
- `--model-name`: 사용할 모델 (spin, spin_h)

### 2. 결측치 보정 결과 저장

학습된 모델을 사용하여 결측치가 보정된 데이터를 STGAN 형식으로 저장할 수 있습니다:

```bash
cd SPIN
python experiments/run_inference_stgan.py --root-dir /path/to/datasets/bay --dataset-name bay_block --model-name spin_h --exp-name YOUR_EXPERIMENT_NAME --output-dir /path/to/save
```

주요 매개변수:

- `--root-dir`: STGAN 데이터셋 경로
- `--dataset-name`: 테스트할 데이터셋 이름
- `--model-name`: 사용한 모델 이름 (spin, spin_h, grin)
- `--exp-name`: 학습 실험 이름 (run_imputation_stgan.py로 학습 시 생성되는 타임스탬프 폴더명)
- `--output-dir`: 보정된 데이터를 저장할 경로
- `--p-fault`, `--p-noise`: 테스트 시 결측치 비율 설정
- `--test-mask-seed`: 테스트 마스크 생성 시드 (여러 개 지정 가능, 예: `--test-mask-seed 1 --test-mask-seed 2`)

생성되는 파일:

- `imputed_data.npy`: 결측치가 보정된 데이터 (STGAN 형식, (시간, 노드, 특성, 채널) 형태)
- `imputation_diff.npy`: 원본과 보정 데이터 간의 차이 (결측치 위치에서만)
- `mask.npy`: 결측치 마스크
- `metadata.txt`: 메타데이터 정보
- `time_features.txt`, `node_adjacent.txt`, `node_dist.txt`: 원본 STGAN 데이터셋에서 복사된 파일

## 주의사항

1. STGAN의 시간 특성 데이터 형식이 TSL에서 기대하는 형식과 다를 수 있으므로, `STGANBayDataset` 클래스의 `datetime_encoded` 메서드를 필요에 따라 조정해야 할 수 있습니다.
2. 이 통합 코드는 STGAN의 Bay 데이터셋에 맞춰 구현되었습니다. 다른 데이터셋을 사용하려면 추가 수정이 필요할 수 있습니다.
3. TSL 라이브러리 버전에 따라 호환성 문제가 발생할 수 있습니다. 호환성 문제가 발생하면 TSL 최신 버전으로 업데이트하거나 코드를 해당 버전에 맞게 수정하세요.
