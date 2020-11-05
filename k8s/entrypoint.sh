#!/usr/bin/env bash
REPO_URL="https://${REPO_USER}@gitlab2.informatik.uni-wuerzburg.de/dmir/dallmann/recommender.git"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"
git clone -q "${REPO_URL}" code
cd code
git checkout -q "${REPO_BRANCH}"
poetry shell
/bin/bash -c "$*"