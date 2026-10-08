# Training image for an NVIDIA DGX (or any Linux host with NVIDIA GPUs).
#
# The CUDA runtime, cuDNN and NCCL come from the locked PyTorch wheels
# (torch 2.10 / CUDA 12.8), so the base image does not need CUDA: the host only
# needs a recent NVIDIA driver (>= 570 recommended for CUDA 12.8) and the
# NVIDIA Container Toolkit (`docker run --gpus ...`).
#
# Build:  docker build -t molpred .
# Run:    see scripts/dgx_train.sh and the README.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# 1) Dependencies only, from the lockfile: cached unless pyproject/uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) The project itself.
COPY src ./src
COPY main.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Data, checkpoints and results are bind-mounted from the host
# (/app/data, /app/outputs, /app/results), so nothing is lost with the container.
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
