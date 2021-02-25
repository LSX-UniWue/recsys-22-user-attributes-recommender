# Trainer

## Checkpoints

TODO

## Loggers

### mlflow
under trainer add a logger section:

```
trainer:
    ...
      logger:
        type: mlflow
        experiment_name: test
        tracking_uri: http://localhost:5000
```