import inspect
from functools import wraps


def _get_hyperparameters(args, kwargs, init_func):
    parameters = list(inspect.signature(init_func).parameters.items())

    hyperparameters = {}

    # first the args

    for index, arg in enumerate(args):
        # index + 1 because self is the first parameter of init
        hyperparameters[parameters[index + 1][0]] = arg

    hyperparameters.update(kwargs)

    # special handling of the model parameter
    model = hyperparameters.get('model', None)
    if model:
        hyperparameters.update(model.hyperparameters)

    return hyperparameters


def save_hyperparameters(init):

    """
    overrides init to set the init values as hyperparameters of the model
    FIXME: currently the module must call self.save_hyperparameters(self.hyperparameters) by itself
    :param init:
    :return:
    """

    @wraps(init)
    def new_init(self, *args, **kwargs):
        self.hyperparameters = _get_hyperparameters(args, kwargs, init)
        init(self, *args, **kwargs)

    return new_init
