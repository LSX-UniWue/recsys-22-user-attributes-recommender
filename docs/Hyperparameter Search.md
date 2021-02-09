# Hyperparameter search

We use Optuna to optimize the hyperparameters of the models in the framework.

## 1. Step

Create a study using your favorite storage backend.

```
optuna create-study --storage $STORAGE_URL
```

Copy the study name from the cli output of optuna.


## 2. Step

Create a run config.

TODO