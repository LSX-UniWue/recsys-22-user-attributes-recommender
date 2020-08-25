FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt update && apt -y install zsh byobu tmux curl htop vim git wget mc && rm -rf /var/lib/apt/lists/*
RUN pip install pytorch-lightning
RUN pip install tqdm pandas 

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

