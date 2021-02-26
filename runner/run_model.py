import logging
import logging.handlers
import typer
import optuna
from optuna.structs import StudyDirection
import json
import _jsonnet
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Dict, Any, Optional
from init.config import Config
from init.container import Container
from init.context import Context
from init.factories.container import ContainerFactory
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import cloud_io

from init.templating.template_engine import TemplateEngine
from init.trainer_builder import CallbackBuilder
from search.processor import ConfigTemplateProcessor
from search.resolver import OptunaParameterResolver

app = typer.Typer()


# FIXME: progress bar is not logged :(
def _config_logging(config: Dict[str, Any]
                    ) -> None:
    logger = logging.getLogger("lightning")
    handler = logging.handlers.RotatingFileHandler(
        Path(config['trainer']['default_root_dir']) / 'run.log', maxBytes=(1048576 * 5), backupCount=7
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_config(config_file: Path) -> Config:
    config_file = Path(config_file)

    if not config_file.exists():
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)

    config_json = _jsonnet.evaluate_file(str(config_file))

    loaded_config = json.loads(config_json)

    config_to_use = TemplateEngine().modify(loaded_config)
    return Config(config_to_use)


def create_container(config: Config) -> Container:
    context = Context()

    container_factory = ContainerFactory()
    container = container_factory.build(config, context)

    return container


def load_container(config_file: Path) -> Container:
    config_raw = load_config(config_file)
    return create_container(config_raw)



@app.command()
def train(config_file: str = typer.Argument(..., help='the path to the config file'),
          do_train: bool = typer.Option(True, help='flag iff the model should be trained'),
          do_test: bool = typer.Option(False, help='flag iff the model should be tested (after training)')
          ) -> None:

    config_file_path = Path(config_file)
    config = load_config(config_file_path)

    container = create_container(config)
    trainer = container.trainer().build()

    if do_train:
        trainer.fit(container.module(),
                    train_dataloader=container.train_dataloader(),
                    val_dataloaders=container.validation_dataloader())

    if do_test:
        if not do_train:
            print(f"The model has to be trained before it can be tested!")
            exit(-1)
        trainer.test(test_dataloaders=container.test_dataloader())


@app.command()
def search(model: str = typer.Argument(..., help="the model to run"),
           template_file: Path = typer.Argument(..., help='the path to the config file'),
           study_name: str = typer.Argument(..., help='the study name of an existing optuna study'),
           study_storage: str = typer.Argument(..., help='the connection string for the study storage'),
           objective_metric: str = typer.Argument(..., help='the name of the metric to watch during the study'
                                                            '(e.g. recall_at_5).'),
           num_trails: int = typer.Option(default=20, help='the number of trails to execute')
          ) -> None:
    # XXX: because the dependency injector does not provide a error message when the config file does not exists,
    # we manually check if the config file exists
    if not os.path.isfile(template_file):
        print(f"the config file cannot be found. Please check the path '{template_file}'!")
        exit(-1)

    def config_from_template(template_file: Path,
                             config_file_handle,
                             trial):
        import yaml
        resolver = OptunaParameterResolver(trial)
        processor = ConfigTemplateProcessor(resolver)

        with template_file.open("r") as f:
            template = yaml.load(f)
            resolved_config = processor.process(template)

            yaml.dump(resolved_config, config_file_handle)
            config_file_handle.flush()

    def objective(trial: optuna.Trial):
        # get the direction to get if we must extract the max or the min value of the metric
        study_direction = trial.study.direction
        objective_best = {StudyDirection.MINIMIZE: min, StudyDirection.MAXIMIZE: max}[study_direction]

        with NamedTemporaryFile(mode='wt') as tmp_config_file:
            config_from_template(template_file, tmp_config_file, trial)

            class MetricsHistoryCallback(Callback):

                def __init__(self):
                    super().__init__()

                    self.metric_history = []

                def on_validation_end(self, trainer, pl_module):
                    self.metric_history.append(trainer.callback_metrics)

            metrics_tracker = MetricsHistoryCallback()

            container = build_container(model, tmp_config_file.name)

            module = container.module()

            config = container.config
            trainer_builder = _get_base_trainer_builder(config.trainer, callbacks=[metrics_tracker])
            trainer = trainer_builder.build()
            trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())

            def _find_best_value(key: str, best: Callable[[List[float]], float] = min) -> float:
                values = [history_entry[key] for history_entry in metrics_tracker.metric_history]
                return best(values)

            return _find_best_value(objective_metric, objective_best)

    study = optuna.load_study(study_name=study_name, storage=study_storage)
    study.optimize(objective, n_trials=num_trails)


@app.command()
def predict(config_file: str = typer.Argument(..., help='the path to the config file'),
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

    container = load_container(Path(config_file))
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

    test_loader = container.test_dataloader()

    callback_params = {
        "output_file_path": output_file,
        "log_input": log_input,
        "tokenizer": container.tokenizer("item"), # FIXME we need to build support for multiple tokenizers
        "strip_padding_tokens": strip_pad_token
    }
    trainer_builder = container.trainer()
    trainer_builder = trainer_builder.add_callback(CallbackBuilder("prediction_logger", callback_params).build())
    trainer_builder = trainer_builder.set("gpus", gpu)
    trainer = trainer_builder.build()

    trainer.test(module, test_dataloaders=test_loader)


@app.command()
def evaluate(config_file: str = typer.Argument(..., help='the path to the config file'),
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

    container = load_container(Path(config_file))
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

    test_loader = container.test_dataloader()

    trainer_builder = container.trainer()
    trainer_builder.set("gpus", gpu)

    trainer = trainer_builder.build()
    trainer.test(module, test_dataloaders=test_loader)


@app.command()
def resume(config_file: str = typer.Argument(..., help='the path to the config file'),
           checkpoint_file: str = typer.Argument(..., help="path to the checkpoint file.")):
    container = load_container(Path(config_file))

    module = container.module()

    trainer_builder = container.trainer()
    trainer = trainer_builder.from_checkpoint(checkpoint_file).build()

    train_loader = container.train_dataloader()
    validation_loader = container.validation_dataloader()

    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    app()
