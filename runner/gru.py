from pyhocon import ConfigFactory
from pytorch_lightning.utilities.cloud_io import load as pl_load
from dm.dota.small import Dota2Small
from modules.gru_module import GRUModule

from runner.util import ConfigTrainerBuilder
from argparse import ArgumentParser

def test(args):
    config_file_path = args.config_file
    # TODO make configurable
    config = ConfigFactory.parse_file(config_file_path)
    datamodule = Dota2Small.from_configuration(config)
    module = GRUModule.from_configuration(config)

    checkpoint_path = config.get_string("checkpoint.path")
    checkpoint = pl_load(f"{checkpoint_path}", map_location=lambda storage, loc: storage)
    module.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer_builder = ConfigTrainerBuilder()
    trainer = trainer_builder.build(config)
    trainer.test(module, datamodule=datamodule)


def train(args):
    config_file_path = args.config_file
    # TODO make configurable
    config = ConfigFactory.parse_file(config_file_path)
    datamodule = Dota2Small.from_configuration(config)
    module = GRUModule.from_configuration(config)

    trainer_builder = ConfigTrainerBuilder()
    trainer = trainer_builder.build(config)
    trainer.fit(module, datamodule=datamodule)


def main():
    parser = ArgumentParser("gru")
    subparser = parser.add_subparsers()
    train_parser = subparser.add_parser("train")
    test_parser = subparser.add_parser("test")

    train_parser.set_defaults(func=train)
    test_parser.set_defaults(func=test)

    train_parser.add_argument("config_file", type=str, help="path to config file")
    test_parser.add_argument("config_file", type=str, help="path to config file")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
