#!/bin/bash

BASE_IMAGE="pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime"
#BASE_IMAGE="pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime"

set -x

container=$(buildah from $BASE_IMAGE)
buildah run $container -- apt-get update
buildah run $container -- apt-get -y install zsh byobu tmux curl htop vim git wget mc
buildah run $container -- rm -rf /var/lib/apt/lists/*
buildah config -e POETRY_HOME=/opt/poetry $container
buildah run $container -- curl --output get-poetry.py https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
buildah run $container -- python get-poetry.py

buildah config -e PATH="/opt/poetry/bin:${PATH}" $container

buildah config -e PYTHONUNBUFFERED=1 $container
buildah config -e LANG C.UTF-8 $container
buildah config -e LC_ALL C.UTF-8 $container

buildah config -e REPO_USER=dallmann $container
buildah config -e REPO_BRANCH=master $container
buildah config -e PROJECT_DIR=/project $container
buildah copy $container .git-askpass /.git-askpass
buildah run $container -- chmod +x /.git-askpass

buildah config -e GIT_TOKEN="123456" $container
buildah config -e GIT_ASKPASS=/.git-askpass $container

buildah copy $container entrypoint.sh /entrypoint.sh
buildah run $container -- chmod ugo+rx /entrypoint.sh

buildah config --entrypoint ["/entrypoint.sh"] $container
