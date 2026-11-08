# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
FROM ${PYTHON_IMAGE} AS build
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SAM2_BUILD_CUDA=0 MMCV_WITH_OPS=1 MAX_JOBS=2 \
    UV_HTTP_TIMEOUT=120 PIP_DEFAULT_TIMEOUT=120
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.9.18
COPY pyproject.toml README.md LICENSE uv.lock ./
COPY requirements/ requirements/
COPY scripts/bootstrap.py scripts/bootstrap.py
COPY src/ src/
RUN python scripts/bootstrap.py --all --root /opt/venvs --cache-dir /tmp/aem-build-cache

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/opt/venvs/app/bin:$PATH" \
    AEM_INFERENCE_CONFIG=/app/configs/inference.docker.toml \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /opt/venvs /opt/venvs
COPY pyproject.toml README.md LICENSE uv.lock ./
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY scripts/ scripts/
COPY docs/ docs/
COPY examples/ examples/
COPY tests/ tests/
COPY requirements/ requirements/
COPY .env.example Dockerfile compose.yaml .dockerignore ./
COPY .github/ .github/
RUN mkdir -p weights data outputs
EXPOSE 8517 8000
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8517", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
