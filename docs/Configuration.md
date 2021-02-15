# Configuration of a model configuration

The configuration for a training contains 4 main config sections:

- Dataset
- Model
- Module
- Trainer


## Module

### Metrics


## Trainer

### Checkpoints

TODO

### Loggers

#### mlflow
under trainer add a logger section:

```
trainer:
    ...
      logger:
        type: mlflow
        experiment_name: test
        tracking_uri: http://localhost:5000
```