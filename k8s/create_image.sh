#!/bin/bash
# base image
BASE_IMAGE="docker.io/pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime"

# user that will be used to pull from the repository
REPO_USER="dallmann"
# secret used to pull from the repository
REPO_SECRET="mysecret"
# branch used to build the image
# TODO automatically determine this value based on the current branch used
REPO_BRANCH="master"

set -x

container=$(buildah from $BASE_IMAGE)

# install packages / clean up
buildah run $container -- apt-get -y update
buildah run $container -- apt-get -y install zsh byobu tmux curl htop vim git wget mc build-essential
buildah run $container -- rm -rf /var/lib/apt/lists/*

# install poetry
buildah config -e POETRY_HOME=/opt/poetry $container
buildah run $container -- curl --output get-poetry.py https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
buildah run $container -- python get-poetry.py -y
#buildah config -e PATH="/opt/poetry/bin:${PATH}" $container

# configure environment
buildah config -e PYTHONUNBUFFERED=1 $container
buildah config -e LANG=C.UTF-8 $container
buildah config -e LC_ALL=C.UTF-8 $container

buildah copy $container ../ /jane-doe-framework
buildah config --workingdir /tmp/jane-doe-framework $container
buildah run $container /opt/poetry/bin/poetry install --no-root --no-dev

#TODO
# determine virtualenv directory,
# adapt permissions
#buildah config -e REPO_USER=dallmann $container
#buildah config -e REPO_BRANCH=master $container
#buildah config -e PROJECT_DIR=/project $container
#buildah copy $container .git-askpass /.git-askpass
#buildah run $container -- chmod +x /.git-askpass
#
#buildah config -e GIT_TOKEN="123456" $container
#buildah config -e GIT_ASKPASS=/.git-askpass $container
#
#buildah copy $container entrypoint.sh /entrypoint.sh
#buildah run $container -- chmod ugo+rx /entrypoint.sh
#
#buildah config --entrypoint ["/entrypoint.sh"] $container
