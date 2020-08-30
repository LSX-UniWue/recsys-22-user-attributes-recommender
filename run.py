from argparse import ArgumentParser

from module_registry import module_registry


def build_module(module_id: str):
    builder = module_registry.get_module_builder(module_id)

    # build argument parser and let the model be build from it
    parser = ArgumentParser()
    parser = builder.add_arguments_to_parser(parser)
    args = parser.parse_args()

    dict_args = vars(args)

    return builder.build_module(dict_args)
