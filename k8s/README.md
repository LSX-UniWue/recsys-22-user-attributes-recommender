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
