# STGAN 이상치 탐지 및 변환 기능

STGAN 모델은 이제 이상치를 탐지하고 NaN 값으로 변환하는 기능을 제공합니다. 이를 통해 이상치 탐지 후 결측치 보간 파이프라인을 구성할 수 있습니다.

## 기능 개요

이 기능은 STGAN 모델이 테스트 과정에서 계산한 MSE(평균 제곱 오차)를 기반으로 이상치를 탐지하고, 선택적으로 이상치를 NaN 값으로 변환합니다. 이는 다음과 같은 단계로 이루어집니다:

1. 모델이 테스트 데이터에 대한 예측 수행
2. 각 데이터 포인트의 MSE 계산
3. 임계값보다 큰 MSE를 가진 데이터 포인트를 이상치로 탐지
4. (선택적) 이상치를 NaN으로 변환한 데이터셋 생성

## 사용 방법

명령줄에서 다음과 같이 인자를 추가하여 이상치 탐지 기능을 사용할 수 있습니다:

```bash
python main.py --dataset bay --outlier_threshold 0.1 --convert_outliers True
```

### 주요 매개변수

- `--outlier_threshold`: 이상치 탐지 임계값(MSE 기준, 기본값: 0.1)
- `--convert_outliers`: 이상치를 NaN으로 변환할지 여부(기본값: False)

## 출력 파일

이상치 탐지 프로세스는 다음 파일들을 생성합니다:

- `outlier_mask.npy`: 이상치 마스크 (1: 이상치, 0: 정상)
- `outlier_mse.npy`: 각 데이터 포인트의 MSE 값
- `outlier_metadata.txt`: 이상치 탐지 관련 메타데이터
- `data_with_nan.npy`: 이상치가 NaN으로 변환된 데이터 (`--convert_outliers True`인 경우)

모든 파일은 `{root_path}/{dataset}/result/` 디렉토리에 저장됩니다.

## 사용 사례: 이상치 탐지 및 보간 파이프라인

이 기능을 활용하여 다음과 같은 파이프라인을 구성할 수 있습니다:

1. STGAN 모델로 이상치 탐지:

   ```bash
   python main.py --dataset bay --outlier_threshold 0.1 --convert_outliers True
   ```
2. 이상치가 NaN으로 변환된 데이터(`data_with_nan.npy`)를 결측치 보간 모델(예: SPIN)에 입력하여 이상치를 보간

이 과정을 통해 데이터에서 이상치를 효과적으로 제거하고 보다 신뢰할 수 있는 값으로 대체할 수 있습니다.

## 주의사항

- 이상치 임계값(`--outlier_threshold`)은 데이터셋 특성에 따라 적절히 조정해야 합니다.
- 이상치 비율이 너무 높거나 낮은 경우 임계값을 재조정하는 것이 좋습니다.
- 생성된 이상치 통계 정보를 확인하여 적절한 임계값을 찾는 것을 권장합니다.
