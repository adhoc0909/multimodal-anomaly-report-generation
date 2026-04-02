# ── API 서버 Dockerfile ────────────────────────────────────────────────────────
# 베이스 이미지: PyTorch 2.5.1 + CUDA 12.1 + Python 3.11
#   - Gemma3 INT4 (로컬 모델) 사용 시 GPU 필요
#   - Gemini API 사용 시 GPU 없어도 동작
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# OpenCV, git 등 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN pip install --no-cache-dir uv

WORKDIR /app

# 의존성 파일 먼저 복사 → 소스 변경 시 레이어 캐시 재사용
COPY pyproject.toml uv.lock ./

# uv sync: uv.lock 기준으로 정확한 버전 설치 (재현 가능한 빌드)
# UV_SYSTEM_PYTHON=1 → .venv 없이 시스템 Python에 직접 설치
# ⚠️ pyproject.toml 패키지 추가/변경 시 로컬에서 `uv lock` 실행 후 커밋 필요
RUN uv sync --frozen

# 소스 전체 복사
COPY . .

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
