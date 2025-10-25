# MyWork

## 설치 및 실행 방법

### 1. 저장소 클론

이 저장소는 Git LFS(Large File Storage)를 사용하여 대용량 파일들을 관리합니다. 따라서 일반적인 `git clone` 대신 다음 명령어를 사용해야 합니다:

```bash
# Git LFS가 설치되어 있지 않은 경우 먼저 설치
sudo apt update && sudo apt install git-lfs -y

# Git LFS 초기화
git lfs install

# 저장소 클론 (LFS 파일들도 함께 다운로드)
git clone https://github.com/KimDoYoung1997/MyWork.git
cd MyWork
```

### 2. 필요한 패키지 설치

```bash
# Python 패키지 설치 (requirements.txt가 있다면)
pip install -r requirements.txt

# 또는 필요한 패키지들을 개별적으로 설치
pip install numpy matplotlib pandas scipy
```

### 3. 실행

```bash
# 메인 스크립트 실행
python my_code_renewal.py
# 또는
python my_code_renewal2.py
```

## Git LFS에 대한 추가 정보

이 저장소는 다음 파일 형식들을 Git LFS로 관리합니다:
- `.dae` - 3D 모델 파일
- `.STL` - STL 메시 파일
- `.obj` - OBJ 메시 파일
- `.npz` - NumPy 압축 데이터 파일
- `.onnx` - ONNX 모델 파일
- `.png` - 이미지 파일

### 일반적인 Git LFS 명령어

```bash
# LFS 파일 상태 확인
git lfs status

# LFS 파일들 수동 다운로드
git lfs pull

# LFS 파일들 수동 업로드
git lfs push origin main
```

## 프로젝트 구조

```
MyWork/
├── config/                 # 설정 파일들
├── modules/               # Python 모듈들
├── npzs/                  # 모션 데이터 파일들
├── performance_plots/     # 성능 분석 플롯들
├── policies/              # 학습된 정책 파일들
├── unitree_description/   # Unitree 로봇 설명 파일들
├── my_code_renewal.py     # 메인 스크립트
└── my_code_renewal2.py    # 대안 스크립트
```
