import logging
import logging.handlers
import typer
import json
import _jsonnet
from pathlib import Path
from typing import Dict, Any, Optional
from init.config import Config
from init.container import Container
from init.context import Context
from init.factories.container import ContainerFactory
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import cloud_io

from runner.util.builder import CallbackBuilder

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


def load_container(config_file: Path) -> Container:
    config_file = Path(config_file)

    if not config_file.exists():
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)

    config_json = _jsonnet.evaluate_file(str(config_file))

    config = Config(json.loads(config_json))
    context = Context()

    container_factory = ContainerFactory()
    container = container_factory.build(config, context)

    return container


@app.command()
def train(config_file: str = typer.Argument(..., help='the path to the config file'),
          do_train: bool = typer.Option(True, help='flag iff the model should be trained'),
          do_test: bool = typer.Option(False, help='flag iff the model should be tested (after training)')
          ) -> None:

    container = load_container(Path(config_file))
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
