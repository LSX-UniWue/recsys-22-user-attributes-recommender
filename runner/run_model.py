import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any

import torch
import typer
from dependency_injector import containers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import cloud_io

from data.datasets import SAMPLE_IDS
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
def run_model(model: str = typer.Argument(..., help="the model to run"),
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
        trainer.test(test_dataloader=container.test_dataloader())

@app.command()
def predict(model: str = typer.Argument(..., help="the model to run"),
            config_file: str = typer.Argument(..., help='the path to the config file'),
            checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file')):

    container = build_container(model)
    container.config.from_yaml(config_file)
    module = container.module()

    # load checkpoint <- we don't use the PL function load_from_checkpoint because it does not work with our module class system
    ckpt = cloud_io.load(checkpoint_file)

    # acquire state_dict
    state_dict = ckpt["state_dict"]

    #TODO remove: is only here because my checkpoint is "old"
    state_dict["model.item_embeddings.embedding.weight"] = state_dict["model.item_embeddings.weight"]
    state_dict.pop("model.item_embeddings.weight", None)

    # load parameters and freeze the model
    module.load_state_dict(state_dict)
    module.freeze()

    # compute logits, softmax and acquire top_k values

    # FIXME need to get correct sample ids through preprocessing on dataset level
    test_loader = container.test_loader()

    # default: disable input output
    log_input = True

    if log_input:
        header = "SID\tRANK\tITEM\tPROBABILITY\tTARGET\tINPUT"
    else:
        header = "SID\tRANK\tITEM\tPROBABILITY\tTARGET"

    print(header)

    for batch_idx, batch in enumerate(test_loader):
        # calculate ids

        sample_ids = batch[SAMPLE_IDS]

        from modules.util.module_util import get_padding_mask
        padding_mask = get_padding_mask(batch["session"], container.tokenizer(), transposed=False, inverse=True)

        logits = module.model(batch["session"], padding_mask)
        probs = torch.softmax(logits, dim=-1)
        values, indices = probs.topk(k=20)

        num_samples_in_batch = indices.size()[0]
        num_values_in_sample = indices.size()[1]

        for in_batch_idx in range(num_samples_in_batch):
            for value_idx in range(num_values_in_sample):
                rank = value_idx + 1
                if log_input:
                    # TODO: remove padding tokens from input
                    print(f"{sample_ids[in_batch_idx]}\t{rank}\t{indices[in_batch_idx][value_idx].item()}\t{values[in_batch_idx][value_idx]}\t{batch['target'][in_batch_idx]}\t{(batch['session'][in_batch_idx].tolist())}")
                else:
                    print(f"{sample_ids[in_batch_idx]}\t{rank}\t{indices[in_batch_idx][value_idx].item()}\t{values[in_batch_idx][value_idx]}\t{batch['target'][in_batch_idx]}")
        return


if __name__ == "__main__":
    app()
