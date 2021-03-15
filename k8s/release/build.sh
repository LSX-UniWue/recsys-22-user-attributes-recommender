#!/usr/bin/env bash
TAG="v0.5.0"
poetry export -o k8s/release/requirements.txt --without-hashes
podman build . --format docker -f k8s/release/Dockerfile -t jane-doe-gpu:${TAG} -t gitlab2.informatik.uni-wuerzburg.de:4567/dmir/dallmann/recommender/jane-doe-gpu:${TAG}
rm k8s/release/requirements.txt