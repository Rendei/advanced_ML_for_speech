# Connect-4 RL lab image. CUDA wheels run on CPU too, so one image serves both.
#   CPU-only build (smaller):  docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t c4rl .
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# Pascal Pons' perfect Connect-4 solver (the ground truth of the evaluation); pinned commit.
ARG SOLVER_REF=d6ba50d8aaf2308c769d9bf2abd42d90f34baf41

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    TOKENIZERS_PARALLELISM=false \
    C4_SOLVER=/opt/connect4/c4solver

# DejaVu for Cyrillic glyphs in the figures; compiler + git to build the solver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core g++ make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/PascalPons/connect4 /opt/connect4 \
    && cd /opt/connect4 && git checkout "${SOLVER_REF}" && make c4solver

WORKDIR /app
RUN pip install torch --index-url "${TORCH_INDEX_URL}"
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY weights ./weights
RUN pip install --no-deps -e .

CMD ["python", "-m", "c4rl", "run", "--profile", "standard"]
