FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    rsync \
    openssh-server \
    tmux \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

COPY architectures/ architectures/
COPY workflow_actions/*.py workflow_actions/
COPY workflow_actions/train/ workflow_actions/train/
COPY workflow_actions/__init__.py workflow_actions/
COPY workflow_actions/dataset_preprocessor/ workflow_actions/dataset_preprocessor/

RUN pip install --no-cache-dir uv \
 && uv sync --frozen

CMD ["bash"]
