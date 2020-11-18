#!/usr/bin/env bash
REPO_URL="https://${REPO_USER}@gitlab2.informatik.uni-wuerzburg.de/dmir/dallmann/recommender.git"
#DATA_REMOTE_PATH=/home/ls6/dallmann/datasets <- put the path in ceph that you need to sync to the ssd into this env variable
#DATA_LOCAL_PATH=/cache <- set this env variable to the ssd cache path
#PREPARE_SCRIPT=somewhere
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"
git clone -q "${REPO_URL}"
cd recommender
git checkout -q "${REPO_BRANCH}"
which poetry
ls -lh `which poetry`
poetry install
chmod +x $PREPARE_SCRIPT
$PREPARE_SCRIPT
$RUN_SCRIPT