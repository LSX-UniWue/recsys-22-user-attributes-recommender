import logging
import logging.handlers
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional

import optuna
import typer
from dependency_injector import containers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import cloud_io

from runner.util.callbacks import PredictionLoggerCallback
from runner.util.containers import BERT4RecContainer, CaserContainer, SASRecContainer, NarmContainer, RNNContainer,\
    DreamContainer
from runner.util.provider_utils import build_standard_logging_callbacks_provider
from search.processor import ConfigTemplateProcessor
from search.resolver import OptunaParameterResolver

app = typer.Typer()


# TODO: introduce a subclass for all container configurations?
def build_container(model_id) -> containers.DeclarativeContainer:
    return {
        'bert4rec': BERT4RecContainer(),
        'sasrec': SASRecContainer(),
        'caser': CaserContainer(),
        "narm": NarmContainer(),
        "rnn": RNNContainer(),
        'dream': DreamContainer()
    }[model_id]


# FIXME: progress bar is not logged :(
def _config_logging(config: Dict[str, Any]
                    ) -> None:
    logger = logging.getLogger("lightning")
    handler = logging.handlers.RotatingFileHandler(
        Path(config['trainer']['default_root_dir']) / 'run.log', maxBytes=(1048576*5), backupCount=7
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@app.command()
def train(model: str = typer.Argument(..., help="the model to run"),
          config_file: str = typer.Argument(..., help='the path to the config file'),
          do_train: bool = typer.Option(True, help='flag iff the model should be trained'),
          do_test: bool = typer.Option(False, help='flag iff the model should be tested (after training)')
          ) -> None:
    # XXX: because the dependency injector does not provide a error message when the config file does not exists,
    # we manually check if the config file exists
    if not os.path.isfile(config_file):
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)

    container = build_container(model)
    container.config.from_yaml(config_file)
    module = container.module()

    trainer = container.trainer()

    # _config_logging(container.config())

    if do_train:
        trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())

    if do_test:
        trainer.test(test_dataloader=container.test_loader())


@app.command()
def search(model: str = typer.Argument(..., help="the model to run"),
           template_file: Path = typer.Argument(..., help='the path to the config file'),
           study_name: str = typer.Argument(..., help='the optuna study name'),
           study_storage: str = typer.Argument(..., help='the connection string for the study storage'),
           objective_metric: str = typer.Argument(..., help='the name of the metric to watch during the study'
                                                            '(e.g. rec_at_5).')
          ) -> None:
    # XXX: because the dependency injector does not provide a error message when the config file does not exists,
    # we manually check if the config file exists
    if not os.path.isfile(template_file):
        print(f"the config file cannot be found. Please check the path '{template_file}'!")
        exit(-1)

    def config_from_template(template_file: Path, config_file_handle, trial):
        import yaml
        resolver = OptunaParameterResolver(trial)
        processor = ConfigTemplateProcessor(resolver)

        with template_file.open("r") as f:
            template = yaml.load(f)
            resolved_config = processor.process(template)

            yaml.dump(resolved_config, config_file_handle)
            config_file_handle.flush()

    def objective(trial):
        with NamedTemporaryFile(mode='wt') as tmp_config_file:
            container = build_container(model)
            config_from_template(template_file, tmp_config_file, trial)

            container.config.from_yaml(tmp_config_file.name)

            module = container.module()

            trainer = container.trainer()
            trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())

            # TODO (AD) We need a way to determine the metrics value for the best checkpoint
            return trainer.callback_metrics[objective_metric]

    study = optuna.load_study(study_name=study_name, storage=study_storage)
    study.optimize(objective, n_trials=20)


@app.command()
def predict(model: str = typer.Argument(..., help="the model to run"),
            config_file: str = typer.Argument(..., help='the path to the config file'),
            checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file'),
            output_file: Path = typer.Argument(..., help='path where output is written'),
            gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
            overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
            log_input: Optional[bool] = typer.Option(default=False, help='enable input logging.'),
            strip_pad_token: Optional[bool] = typer.Option(default=True, help='strip pad token, if input is logged.')
            ):

    if not overwrite and output_file.exists():
        print(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(-1)

    container = build_container(model)
    container.config.from_yaml(config_file)
    module = container.module()

    # FIXME: try to use load_from_checkpoint later
    # load checkpoint <- we don't use the PL function load_from_checkpoint because it does
    # not work with our module class system
    ckpt = cloud_io.load(checkpoint_file)

    # acquire state_dict
    state_dict = ckpt["state_dict"]

    # load parameters and freeze the model
    module.load_state_dict(state_dict)
    module.freeze()

    test_loader = container.test_loader()

    callbacks = [
        PredictionLoggerCallback(output_file_path=output_file,
                                 log_input=log_input,
                                 tokenizer=container.tokenizer(),
                                 strip_padding_tokens=strip_pad_token)
    ]
    trainer = Trainer(callbacks=callbacks, gpus=gpu)
    trainer.test(module, test_dataloaders=test_loader)


#FIXME: (AD) metrics are not calculated correctly
#FIXME: (AD) need to write output to file, but first need to resolve Exception caused by trainer test loop :-/
@app.command()
def evaluate(model: str = typer.Argument(..., help="the model to run"),
             config_file: str = typer.Argument(..., help='the path to the config file'),
             checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file'),
             output_file: Path = typer.Argument(..., help='path where output is written'),
             gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
             overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
             seed: Optional[int] = typer.Option(default=42, help='seed for rng')
             ):

    if not overwrite and output_file.exists():
        print(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(-1)

    seed_everything(seed)

    container = build_container(model)
    container.config.from_yaml(config_file)
    module = container.module()

    # FIXME: try to use load_from_checkpoint later
    # load checkpoint <- we don't use the PL function load_from_checkpoint because it does
    # not work with our module class system
    ckpt = cloud_io.load(checkpoint_file)

    # acquire state_dict
    state_dict = ckpt["state_dict"]

    # load parameters and freeze the model
    module.load_state_dict(state_dict)
    module.freeze()

    test_loader = container.test_loader()

    callbacks = build_standard_logging_callbacks_provider(container.config.module)()

    trainer = Trainer(gpus=gpu, callbacks=callbacks)
    trainer.test(module, test_dataloaders=test_loader)


if __name__ == "__main__":
    app()
