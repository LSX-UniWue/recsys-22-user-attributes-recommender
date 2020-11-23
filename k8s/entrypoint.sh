#!/usr/bin/env bash
REPO_URL="https://${REPO_USER}@gitlab2.informatik.uni-wuerzburg.de/dmir/dallmann/recommender.git"
#DATA_REMOTE_PATH=/home/ls6/dallmann/datasets <- put the path in ceph that you need to sync to the ssd into this env variable
#DATA_LOCAL_PATH=/cache <- set this env variable to the ssd cache path
#PREPARE_SCRIPT=somewhere
#RUN_SCRIPT
# create project dir
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"
# checkout the repository
git clone -q "${REPO_URL}"
cd recommender || exit 1
git checkout -q "${REPO_BRANCH}"

# install requirements using poetry
ls -lh `which poetry`
poetry install

# execute the provided prepare script
if [ ! -z "$PREPARE_SCRIPT" ]; then
  chmod +x "${PREPARE_SCRIPT}"
  $PREPARE_SCRIPT
fi

# now run the configuration using poetry
/bin/bash -c "cd ${PROJECT_DIR}/recommender && poetry run python -m runner.run_model $*"