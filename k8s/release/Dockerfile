FROM docker.io/pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG TAG=latest
ENV WHEEL_FILENAME="asme-${TAG}-py3-none-any.whl"

# install some basic utilities
RUN apt-get update && apt-get -y install zsh byobu tmux curl htop vim git wget mc build-essential && rm -rf /var/lib/apt/lists/*

# install poetry into /opt/poetry
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="/opt/poetry/bin:${PATH}"
RUN chmod o+rx /opt/poetry/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install --no-cache-dir torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

COPY dist/$WHEEL_FILENAME /
RUN pip install --no-cache-dir /$WHEEL_FILENAME

COPY /k8s/release/entrypoint.sh /entrypoint.sh

RUN chmod ugo+rx /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh" ]
