#!/usr/bin/env bash
REPO_URL="https://${REPO_USER}@gitlab2.informatik.uni-wuerzburg.de/dmir/dallmann/recommender.git"
#DATA_REMOTE_PATH=/home/ls6/dallmann/datasets <- put the path in ceph that you need to sync to the ssd into this env variable
#DATA_LOCAL_PATH=/cache <- set this env variable to the ssd cache path
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"
git clone -q "${REPO_URL}" code
cd code
git checkout -q "${REPO_BRANCH}"
poetry install
cp -r $DATA_REMOTE_PATH $DATA_LOCAL_PATH
/bin/bash -c "$*"