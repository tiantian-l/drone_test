# Slim GPU image for drone-dreamer-nav (only the deps this project needs).
# Build & run:
#   docker build -f cloud/Dockerfile -t drone-dreamer-nav .
#   docker run --gpus all -it --rm -v ~/logdir:/logdir drone-dreamer-nav \
#     --configs drone_nav --logdir /logdir/{timestamp} --jax.platform cuda
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y \
    python3.11-dev python3.11-venv ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools wheel

WORKDIR /app
COPY . .

# numpy<2 pinned first (DreamerV3 requirement), then the drone env + DreamerV3.
RUN pip install "numpy<2" \
    && pip install -e third_party/gym-pybullet-drones \
    && grep -v '^jax' third_party/dreamerv3/requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt \
    && pip install "jax[cuda12]==0.4.33"

# drone_nav at repo root must be importable from third_party/dreamerv3.
ENV PYTHONPATH="/app"
WORKDIR /app/third_party/dreamerv3
ENTRYPOINT ["python", "dreamerv3/main.py"]
