# Trainer


## Early Stopping:

If you want to early stop training based on a metric or the loss, add the following to the trainer config:

```
trainer: {
    ...
    early_stopping: {
      monitor: METRIC_OR_LOSS,
      min_delta: MIN_DELTA,
      patience: PATIENCE,
      mode: MODE
    },
    ...
}
```

See [Pytorch Lightning Docu](https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html?highlight=early%20stopping#early-stopping-based-on-metric-using-the-earlystopping-callback) for more details.

## Checkpoints

TODO

## Loggers

### mlflow
under trainer add a logger section:

```
trainer: {
    ...
      logger: {
        type: mlflow,
        experiment_name: test,
        tracking_uri: http://localhost:5000
    }
    ...
}
```