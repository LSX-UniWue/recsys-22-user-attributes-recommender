from pytorch_lightning import Callback


# TODO: Implement as soon as we migrated to PL 1.1.0. This version of PL will provide the 'on_before_zero_grad' callback.
class GradientLoggerCallback(Callback):
    pass