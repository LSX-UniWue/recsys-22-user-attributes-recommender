FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

## config
ARG USER=dallmann
ARG UID=1038

## setup (you don't have to touch this)
RUN touch `date` && apt-get update && apt-get upgrade

RUN apt-get install -y python3 python3-pip python3-dev python3-venv zsh byobu tmux curl htop vim git wget mc

RUN adduser ${USER} --uid ${UID} --home /home/ls6/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/ls6/${USER}
RUN chown -R ${USER} /home/ls6/${USER}


COPY ../../requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
USER ${UID}

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

USER ${UID}
