FROM python:3.10.12-slim-bullseye

ENV HOME=/workspace \
    LD_LIBRARY_PATH=/usr/local/lib/ \
    PIP_NO_CACHE_DIR="off"

ARG DEBIAN_FRONTEND="noninteractive"

EXPOSE 6006 8888 5000

WORKDIR $HOME

RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    screen \
    openssh-client \
    zsh \
    less \
    git \
    wget \
    curl \
    libeccodes-dev \
    libgeos-dev \
    libgdal-dev \
    build-essential \
    libffi-dev \
    openjdk-17-jdk \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python libraries
COPY poetry.lock pyproject.toml ${HOME}/

RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root
