import os

import typer
from dependency_injector import containers

from runner.util.containers import BERT4RecContainer, CaserContainer, SASRecContainer, NarmContainer, GRUContainer


app = typer.Typer()


# TODO: introduce a subclass for all container configurations?
def build_container(model_id) -> containers.DeclarativeContainer:
    return {
        'bert4rec': BERT4RecContainer(),
        'sasrec': SASRecContainer(),
        'caser': CaserContainer(),
        "narm": NarmContainer(),
        "gru": GRUContainer()
    }[model_id]


@app.command()
def run_model(model: str = typer.Argument(..., help="the model to run"),
              config_file: str = typer.Argument(..., help='the path to the config file')
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
    trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())


if __name__ == "__main__":
    app()
