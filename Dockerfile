FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    rsync \
    openssh-server \
    tmux \
    nano \
    make \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install --no-cache-dir uv && \
    uv sync --frozen

ENV PYTHONPATH=/app

COPY Makefile .
COPY architectures/ architectures/
COPY workflow_actions/ workflow_actions/

RUN make prepare_dirs

RUN touch /root/.no_auto_tmux

CMD ["bash"]
