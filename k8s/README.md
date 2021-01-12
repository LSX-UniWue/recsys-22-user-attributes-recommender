## Build
```
export BUILDAH_LAYERS=true
buildah bud --format docker -f Dockerfile -t lsx-harbor.informatik.uni-wuerzburg.de/dallmann/recommender:latest -t recommender:latest .
```

## Usage
### Generate Gitlab access token
Even though the password is supplied via an environment variable that can be populated by a kubernetes secret, you
probably don't want to share it with the admins. Lucky for us, we can also generate a gitlab access token for that
purpose. 

1. Go to your Profile and select Settings->Access Tokens
2. Give it a name, e.g. `k8s`
3. Select `read_repository`
4. Save the final token, because it won't be accessible afterwards

### Create kubernetes secret

```
kubectl -n <namespace> create secret generic gitlab-token --from-literal=token=xxx
```

## Run on your local machine
```
podman run -e GIT_TOKEN=S369mz5Ur1hS2KFBxiDH -e REPO_USER=ald58ab -e REPO_BRANCH=container --mount=type=bind,source=/home/dallmann/uni/research/dota/datasets/small/preprocessed/match_split,dst=/dataset --mount=type=bind,source=/home/dallmann/tmp/configs,dst=/configs recommender:latest poetry run python -m runner.run_model narm /configs/narm_config.yaml
```