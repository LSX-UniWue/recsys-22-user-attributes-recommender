# Build Container Images
## Development
### Build
```
cd k8s/dev
podman build . --format docker -f Dockerfile -t lsx-harbor.informatik.uni-wuerzburg.de/dallmann/recommender/jane-doe-gpu:latest -t jane-doe-gpu:latest
```
### Run
```shell
podman run -e GIT_TOKEN=<token> -e REPO_USER=<user> -e REPO_BRANCH=container --mount=type=bind,source=/home/dallmann/uni/research/dota/datasets/small/preprocessed/match_split,dst=/dataset --mount=type=bind,source=/home/dallmann/tmp/configs,dst=/configs recommender:latest poetry run python -m runner.run_model narm /configs/narm_config.yaml
```
### Environment Variables
* PREPARE_SCRIPT:
     If set the script will be executed before the framework command is run. This is useful for copying data to a faster drive
* GIT_TOKEN: 
     Token used to checkout the repository
* REPO_USER:
     User used to checkout the repository
* REPO_BRANCH:
     Branch to checkout
* PROJECT_DIR:
     A writeable path within the container, that will be used to store the repository

## Release
### Build
```shell
bash k8s/release/build.sh
```
### Run
```shell
podman run --mount=type=bind,source=<local path to dataset>,dst=/dataset --mount=type=bind,source=<local path to configuraton files>,dst=/configs jane-do-framework:<TAG> <model> /configs/<config>.jsonnet
```
### Environment Variables
* PREPARE_SCRIPT: if set the script will be executed before the framework command is run. This is useful for copying data to a faster drive.

### Kubernetes
Both containers can be used as in a kubernetes cluster. Please follow the instructions below to correctly setup your
namespace for the `Development` container.

#### Development
The development container does not come with a preinstalled version of the framework. Instead, the entrypoint will clone
the specified branch and setup an environment using poetry. Your command will be executed inside this environment.

TODO: add documentation for entrypoint hooks

##### Generate Gitlab access token
Even though the password is supplied via an environment variable that can be populated by a kubernetes secret, you
probably don't want to share it with the admins. Lucky for us, we can also generate a gitlab access token for that
purpose. 

1. Go to your Profile and select Settings->Access Tokens
2. Give it a name, e.g. `k8s`
3. Select `read_repository`
4. Save the final token, because it won't be accessible afterwards

##### Create kubernetes secret

```
kubectl -n <namespace> create secret generic gitlab-token --from-literal=token=xxx
```

