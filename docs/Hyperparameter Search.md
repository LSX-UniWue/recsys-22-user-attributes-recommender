# Hyperparameter search

We use Optuna to optimize the hyperparameters of the models in the framework.

## 1. Step

Create a study using your favorite storage backend.

```
optuna create-study --study-name $STUDY_NAME --storage $STORAGE --direction $DIRECTION
```

e.g. storage can be `redis://redis`, direction can be `{minimize,maximize}` and should be `maximize` for recall@k

Copy the study name from the cli output of optuna.


## 2. Step

Create a run config.

TODO