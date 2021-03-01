# Trainer

## Checkpoints

TODO

## Loggers

### mlflow
Under trainer add a logger section:

```
trainer:
    ...
      logger:
        type: mlflow
        experiment_name: test
        tracking_uri: http://localhost:5000
```

### wandb
Under trainer add a logger section:
```
trainer {
    ...
      logger: {
        type: "wandb",
        log_model: false,
        project: "test"
      },
    ...
}
```