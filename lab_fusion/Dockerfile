# Fusion lab image. CUDA wheels run on CPU too, so one image serves both.
#   CPU-only build (smaller):  docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t fusionlab .
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/app/cache/hf \
    TOKENIZERS_PARALLELISM=false

# libsndfile for soundfile, DejaVu for Cyrillic glyphs in the figures.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch --index-url "${TORCH_INDEX_URL}"
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-deps -e .

CMD ["python", "-m", "fusionlab", "run", "--profile", "standard"]
