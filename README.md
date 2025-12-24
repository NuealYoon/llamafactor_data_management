# 🦙 LLaMA Factory Data Management System

LLaMA Factory를 활용한 UI 기반 모델 학습 및 데이터 관리 시스템입니다. Gradio와 PostgreSQL을 사용하여 직관적인 웹 인터페이스를 통해 데이터셋 관리, 모델 학습, 학습된 모델 관리를 할 수 있습니다.

## ✨ 주요 기능

### 📊 데이터셋 관리
- 다양한 형식의 데이터셋 생성 (Alpaca, ShareGPT, Custom)
- 데이터셋 샘플 추가/편집/삭제
- JSON 파일로 데이터셋 가져오기/내보내기
- 데이터셋 통계 및 미리보기

### 🚀 학습 관리
- 학습 작업 생성 및 설정
- 실시간 학습 모니터링
- 학습 로그 확인
- 학습 작업 시작/중지
- LoRA 파라미터 세밀 조정

### 🤖 모델 관리
- 학습된 모델 등록 및 관리
- 모델 메타데이터 관리
- 모델 활성화/비활성화
- 모델 삭제 (파일 포함)

## 🏗️ 프로젝트 구조

```
llamafactor_data_management/
├── app/
│   ├── database/           # 데이터베이스 모델 및 CRUD
│   │   ├── models.py       # SQLAlchemy 모델
│   │   ├── database.py     # DB 연결
│   │   └── crud.py         # CRUD 작업
│   ├── services/           # 비즈니스 로직
│   │   ├── dataset_service.py
│   │   ├── training_service.py
│   │   └── model_service.py
│   ├── ui/                 # Gradio UI 컴포넌트
│   │   ├── dataset_ui.py
│   │   ├── training_ui.py
│   │   └── model_ui.py
│   └── main.py             # 메인 애플리케이션
├── config/                 # 설정 파일
│   └── config.py
├── data/                   # 데이터 저장소
│   └── datasets/
├── models/                 # 모델 저장소
│   └── trained/
├── requirements.txt        # Python 패키지
├── docker-compose.yml      # Docker 설정
├── .env.example            # 환경 변수 예제
└── README.md
```

## 🚀 빠른 시작

### 1. 사전 요구사항

- Python 3.10+
- PostgreSQL 12+
- CUDA (GPU 학습 시)
- Docker & Docker Compose (선택사항)

### 2. 설치

#### 방법 1: Docker Compose 사용 (권장)

```bash
# 저장소 클론
git clone <repository-url>
cd llamafactor_data_management

# 환경 변수 설정
cp .env.example .env

# PostgreSQL 시작
docker-compose up -d postgres

# Python 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 애플리케이션 실행
python app/main.py
```

#### 방법 2: 로컬 PostgreSQL 사용

```bash
# PostgreSQL 설치 (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# 데이터베이스 생성
sudo -u postgres psql
CREATE DATABASE llama_factory_db;
CREATE USER postgres WITH PASSWORD 'postgres';
GRANT ALL PRIVILEGES ON DATABASE llama_factory_db TO postgres;
\q

# 나머지는 방법 1과 동일
cp .env.example .env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

### 3. 접속

브라우저에서 다음 주소로 접속:
- **메인 애플리케이션**: http://localhost:7860
- **PgAdmin** (Docker 사용 시): http://localhost:5050
  - Email: admin@admin.com
  - Password: admin

## ⚙️ 설정

### 환경 변수 (.env)

```bash
# 데이터베이스 설정
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=llama_factory_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# 애플리케이션 설정
APP_HOST=0.0.0.0
APP_PORT=7860
DEBUG=True

# 경로 설정
BASE_MODEL_PATH=./models/base
TRAINED_MODEL_PATH=./models/trained
DATASET_PATH=./data/datasets

# 학습 설정
MAX_CONCURRENT_TRAININGS=2
DEFAULT_BATCH_SIZE=4
DEFAULT_LEARNING_RATE=5e-5

# 로깅
LOG_LEVEL=INFO
WANDB_API_KEY=your_wandb_key  # 선택사항
```

## 📖 사용 방법

### 데이터셋 생성 및 관리

1. **Dataset Management** 탭으로 이동
2. 새 데이터셋 생성:
   - 데이터셋 이름 입력
   - 타입 선택 (Alpaca/ShareGPT/Custom)
   - 설명 입력 (선택사항)
   - "Create Dataset" 클릭
3. 샘플 추가:
   - 데이터셋 선택
   - Instruction, Output 입력
   - Input, System Prompt 입력 (선택사항)
   - "Add Sample" 클릭

### JSON 파일에서 데이터 가져오기

데이터셋 JSON 형식:
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris.",
    "system": "You are a helpful assistant."
  }
]
```

### 모델 학습

1. **Training Management** 탭으로 이동
2. 새 학습 작업 생성:
   - 작업 이름 입력
   - 데이터셋 선택
   - 베이스 모델 경로 입력 (예: `meta-llama/Llama-2-7b-hf`)
   - 학습 파라미터 설정
   - "Create Training Job" 클릭
3. 학습 시작:
   - Job ID 입력
   - "▶️ Start" 클릭

### 학습 파라미터

- **Learning Rate**: 학습률 (기본: 5e-5)
- **Epochs**: 학습 에포크 수 (기본: 3)
- **Batch Size**: 배치 크기 (기본: 4)
- **Gradient Accumulation**: 그래디언트 누적 단계 (기본: 8)
- **Max Sequence Length**: 최대 시퀀스 길이 (기본: 512)
- **LoRA Rank**: LoRA 랭크 (기본: 8)
- **LoRA Alpha**: LoRA 알파 (기본: 16)
- **LoRA Dropout**: LoRA 드롭아웃 (기본: 0.05)

### 학습된 모델 관리

1. **Model Management** 탭으로 이동
2. 모델 등록:
   - Training Job ID 입력
   - 모델 이름 입력
   - "Register Model" 클릭
3. 모델 조회 및 관리:
   - 등록된 모델 목록 확인
   - 모델 활성화/비활성화
   - 모델 삭제

## 🔧 개발

### 데이터베이스 스키마

주요 테이블:
- **datasets**: 데이터셋 정보
- **dataset_samples**: 데이터셋 샘플
- **training_jobs**: 학습 작업
- **trained_models**: 학습된 모델

### API 구조

```python
# 데이터셋 서비스
DatasetService.create_dataset(db, name, type, description)
DatasetService.add_sample(db, dataset_id, instruction, output, ...)
DatasetService.import_from_json(db, dataset_id, file_path)

# 학습 서비스
TrainingService.create_training_job(db, name, dataset_id, base_model, ...)
TrainingService.start_training(db, job_id)
TrainingService.stop_training(db, job_id)

# 모델 서비스
ModelService.create_model_from_training(db, training_job_id, name, ...)
ModelService.get_models(db, is_active)
```

## 🐛 문제 해결

### PostgreSQL 연결 오류
```bash
# PostgreSQL 서비스 상태 확인
sudo systemctl status postgresql

# Docker 컨테이너 확인
docker-compose ps
docker-compose logs postgres
```

### 패키지 설치 오류
```bash
# CUDA 관련 오류 시 (CPU 전용 사용)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# psycopg2 설치 오류 시
sudo apt-get install libpq-dev python3-dev
```

### 데이터베이스 초기화
```bash
# 데이터베이스 리셋
docker-compose down -v
docker-compose up -d postgres
python app/main.py  # 자동으로 테이블 생성
```

## 📝 TODO

- [ ] LLaMA Factory 실제 학습 통합
- [ ] 실시간 학습 진행률 모니터링
- [ ] 모델 추론 인터페이스
- [ ] 멀티 GPU 학습 지원
- [ ] 데이터셋 버전 관리
- [ ] 학습 결과 시각화 (TensorBoard, WandB)
- [ ] 모델 평가 및 비교 기능
- [ ] API 엔드포인트 추가 (FastAPI)

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 📞 문의

문제가 있거나 질문이 있으시면 이슈를 등록해주세요.
