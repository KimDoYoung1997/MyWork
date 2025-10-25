# MyWork

## 설치 및 실행 방법

### 1. 저장소 클론

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
# TODO

```

### 3. 실행

```bash
# 메인 스크립트 실행
python my_code_renewal.py
# 또는
python my_code_renewal2.py
```


## 프로젝트 구조

```
MyWork/
├── config/                # 설정 파일
├── modules/               # Python 모듈
├── npzs/                  # 모션 데이터 파일
├── performance_plots/     # 성능 분석 플롯
├── policies/              # 학습된 정책 파일
├── unitree_description/   # Unitree 로봇 파일
├── my_code_renewal.py     # 메인 스크립트
└── my_code_renewal2.py    # 대안 스크립트
```
