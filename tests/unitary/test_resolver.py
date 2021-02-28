import optuna
import pytest
from optuna import Trial

from init.templating.search import OptunaParameterResolver


@pytest.fixture
def base_resolver():
    study = optuna.create_study(optuna.storages.InMemoryStorage())
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    return OptunaParameterResolver(trial)


@pytest.fixture
def basic_config():
    return {
        "suggest_int":
            {
                "name": "x",
                "low": 8,
                "high": 64
            }
    }


def test_resolve(base_resolver, basic_config):
    resolved_value = base_resolver.resolve(basic_config)
    assert type(resolved_value) == int
