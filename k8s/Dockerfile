FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

## config
ARG USER=dallmann
ARG UID=1038

## setup (you don't have to touch this)
RUN touch `date` && apt-get update

RUN apt install python3 python3-pip python3-dev python3-venv zsh byobu tmux curl htop vim git wget mc -y

RUN adduser ${USER} --uid ${UID} --home /home/ls6/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/ls6/${USER}
RUN chown -R ${USER} /home/ls6/${USER}

RUN pip3 install tqdm numpy scipy sklearn pandas torch torchvision pytorch-lightning
USER ${UID}

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

USER ${UID}
