# Hyperparameter Search

We use Optuna to optimize the hyperparameters of the models in the framework.
Optional you can use MLFlow to track your current parameter search.

## 1. Step: Setup Infrastructure

### Optuna Storage

Setup a storage for optuna.

There are kubernetes examples for redis in this repo.

### MLFlow

Setup mflow, configure it in the run config

## 2. Step: Create Optuna Study

Create a study using your favorite storage backend.

```
optuna create-study --study-name $STUDY_NAME --storage $STORAGE --direction $DIRECTION
```

e.g. storage can be `redis://redis`, direction can be `{minimize,maximize}` and should be `maximize` for recall@k

Copy the study name from the cli output of optuna.


## 3. Step: Create Run Config

Create a run config.


## 4. Step: Run Study