import os
from argparse import ArgumentParser

from dependency_injector import containers

from runner.util.containers import BERT4RecContainer, CaserContainer, SASRecContainer, NarmContainer, GRUContainer


# TODO: introduce a subclass for all container configurations?
def build_container(model_id) -> containers.DeclarativeContainer:
    return {
        'bert4rec': BERT4RecContainer(),
        'sasrec': SASRecContainer(),
        'caser': CaserContainer(),
        "narm": NarmContainer(),
        "gru": GRUContainer()
    }[model_id]


def run_model(model_id: str,
              path_to_config_file: str
              ) -> None:
    container = build_container(model_id)
    container.config.from_yaml(path_to_config_file)
    module = container.module()

    trainer = container.trainer()
    trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model", help="the model to train", type=str)
    parser.add_argument("path_to_config_file", help="the path to the config file", type=str)

    args = parser.parse_args()

    # XXX: because the dependency injector does not provide a error message when the config file does not exists,
    # we manually check if the config file exists
    config_file = args.path_to_config_file
    if not os.path.isfile(config_file):
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)

    run_model(args.model, config_file)
