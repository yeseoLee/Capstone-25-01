# STGAN_tsl: TSL PemsBay 데이터셋을 위한 STGAN

이 저장소는 TSL(Torch Spatiotemporal Library)의 PemsBay 데이터셋을 사용하여 STGAN(Spatiotemporal Graph Attention Network)을 학습하고 테스트하는 코드를 제공합니다.

## 개요

STGAN_tsl은 다음 기능을 제공합니다:

1. TSL의 PemsBay 데이터셋 로드 및 전처리
2. PemsBay 데이터셋을 STGAN 형식으로 변환
3. 결측치가 있는 PemsBay 데이터셋 생성 (블록 결측치와 포인트 결측치)
4. STGAN 모델을 사용한 결측치 보간 및 성능 평가

## 설치 방법

1. 필요한 라이브러리 설치:

```bash
# TSL 설치
pip install torch-spatiotemporal-library

# STGAN 필요 패키지 설치
pip install numpy pandas torch matplotlib
```

2. 저장소 클론:

```bash
git clone <repository_url>
cd <repository_directory>
```

## 파일 구조

- `load_pemsbay.py`: PemsBay 데이터셋을 로드하고 결측치를 추가하는 함수 제공
- `convert_format.py`: PemsBay 데이터셋을 STGAN 형식으로 변환하는 함수 제공
- `run_stgan_pemsbay.py`: STGAN 모델을 학습하고 평가하는 실험 스크립트

## 사용 방법

### 1. PemsBay 데이터셋 로드 및 결측치 추가

```bash
python STGAN_tsl/load_pemsbay.py
```

이 스크립트는 PemsBay 데이터셋을 로드하고 블록 결측치와 포인트 결측치를 추가한 두 가지 버전의 데이터셋을 생성합니다.

### 2. PemsBay 데이터셋을 STGAN 형식으로 변환

```bash
python STGAN_tsl/convert_format.py
```

이 스크립트는 PemsBay 데이터셋을 STGAN에서 사용할 수 있는 형식으로 변환합니다. 변환된 데이터는 `STGAN_tsl/data/stgan_format_*` 디렉토리에 저장됩니다.

### 3. STGAN 모델 학습 및 평가

```bash
# 블록 결측치에 대한 실험
python STGAN_tsl/run_stgan_pemsbay.py --missing_type block --epochs 100 --batch_size 64

# 포인트 결측치에 대한 실험
python STGAN_tsl/run_stgan_pemsbay.py --missing_type point --epochs 100 --batch_size 64
```

이 스크립트는 결측치가 있는 PemsBay 데이터셋을 사용하여 STGAN 모델을 학습하고 평가합니다. 학습된 모델은 `STGAN_tsl/models/` 디렉토리에 저장되고, 평가 결과는 `STGAN_tsl/results/` 디렉토리에 저장됩니다.

### 4. 학습된 모델 테스트

```bash
# 이미 학습된 모델 테스트
python STGAN_tsl/run_stgan_pemsbay.py --missing_type block --test_only
```

`--test_only` 플래그를 사용하면 모델 학습 없이 테스트만 수행합니다.

## 주요 매개변수

### 결측치 생성

- `p_fault`: 연속적인 결측치(블록)를 생성하는 확률 (기본값: 0.0015)
- `p_noise`: 독립적인 결측치(포인트)를 생성하는 확률 (기본값: 0.05, 포인트 유형: 0.25)
- `seed`: 랜덤 시드

### STGAN 모델

- `time_steps`: 시간 창 크기 (기본값: 12)
- `node_steps`: 노드 스텝 크기 (기본값: 5)
- `node_dim`: 노드 임베딩 차원 (기본값: 40)
- `dropout`: 드롭아웃 비율 (기본값: 0.1)
- `attention`: 어텐션 메커니즘 사용 여부
- `time_bn`: 시간 배치 정규화 사용 여부
- `channel_bn`: 채널 배치 정규화 사용 여부

### 학습 관련

- `epochs`: 학습 에폭 수 (기본값: 100)
- `lr`: 학습률 (기본값: 0.001)
- `batch_size`: 배치 크기 (기본값: 64)
- `verbose_iter`: 학습 상태 출력 간격 (기본값: 10)

## 참고 문헌

- TSL(Torch Spatiotemporal Library): https://torch-spatiotemporal.readthedocs.io/
- PemsBay 데이터셋: California Transportation Performance Measurement System
- STGAN: Spatiotemporal Graph Attention Network

## 라이센스

MIT 