import os

import numpy as np
import typer
import optuna
import json

from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional, Callable, List, Iterator
from optuna.study import StudyDirection
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.utilities import cloud_io
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.dataset import T_co

from data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, TARGET_ENTRY_NAME
from asme.init.context import Context
from asme.init.factories.metrics.metrics_container import MetricsContainerFactory
from asme.init.templating.search.configuration import SearchConfigurationTemplateProcessor
from asme.init.templating.search.processor import SearchTemplateProcessor
from asme.init.templating.search.resolver import OptunaParameterResolver
from asme.utils.run_utils import load_config, create_container, load_container
from asme.tokenization.tokenizer import Tokenizer
from asme.utils import ioutils
from asme.utils.ioutils import load_file_with_item_ids, determine_log_dir, save_config, save_finished_flag, \
    finished_flag_exists
from asme.writer.prediction.prediction_writer import build_prediction_writer
from asme.writer.results.results_writer import build_result_writer


app = typer.Typer()


@app.command()
def train(config_file: str = typer.Argument(..., help='the path to the config file'),
          do_train: bool = typer.Option(True, help='flag iff the model should be trained'),
          do_test: bool = typer.Option(False, help='flag iff the model should be tested (after training)')
          ) -> None:
    if do_test and not do_train:
        print(f"The model has to be trained before it can be tested!")
        exit(-1)

    config_file_path = Path(config_file)
    config = load_config(config_file_path)

    container = create_container(config)
    trainer = container.trainer().build()

    # save plain json config to the log dir/root dir of the trainer
    log_dir = Path(determine_log_dir(trainer))
    save_config(config, log_dir)

    if do_train:
        trainer.fit(container.module(),
                    train_dataloader=container.train_dataloader(),
                    val_dataloaders=container.validation_dataloader())

    save_finished_flag(log_dir)

    if do_test:
        trainer.test(test_dataloaders=container.test_dataloader())


@app.command()
def search(template_file: Path = typer.Argument(..., help='the path to the config file'),
           study_name: str = typer.Argument(..., help='the study name of an existing optuna study'),
           study_storage: str = typer.Argument(..., help='the connection string for the study storage'),

           objective_metric: str = typer.Argument(..., help='the name of the metric to watch during the study'
                                                            '(e.g. recall@5).'),
           study_direction: str = typer.Option(default="maximize", help="minimize / maximize"),
           num_trails: int = typer.Option(default=20, help='the number of trails to execute')
           ) -> None:
    # check if objective_metric is defined
    test_config = load_config(template_file)
    test_metrics_config = test_config.get_config(['module', 'metrics'])

    metrics_factors = MetricsContainerFactory()
    test_metrics_container = metrics_factors.build(test_metrics_config, Context())
    if objective_metric not in test_metrics_container.get_metric_names():
        raise ValueError(f'{objective_metric} not configured. '
                         f'Can not optimize hyperparameters using the specified objective')

    def config_from_template(template_file: Path,
                             config_file_handle: NamedTemporaryFile,
                             trial: optuna.Trial):
        """
        Loads the template file and applies all template processors (including the SearchTemplateProcessor). The result
        is written to the temporary `config_file_handle`.

        :param template_file: a config file template path.
        :param config_file_handle: a file handle for the temporary config file.
        :param trial: a trial object.
        :return: nothing.
        """
        config = load_config(template_file, [SearchTemplateProcessor(OptunaParameterResolver(trial))],
                             [SearchConfigurationTemplateProcessor(trial)])
        json.dump(config.config, config_file_handle)
        config_file_handle.flush()

    # FIXME (AD) don't rely on capturing of arguments in internal function scope -> make it into a callable object
    def objective(trial: optuna.Trial):
        # get the direction to get if we must extract the max or the min value of the metric
        trail_study_direction = trial.study.direction
        objective_best = {
            StudyDirection.MINIMIZE: min,
            StudyDirection.MAXIMIZE: max
        }[trail_study_direction]

        with NamedTemporaryFile(mode='wt') as tmp_config_file:
            # load config template, apply processors and write to `tmp_config_file`
            config_from_template(template_file, tmp_config_file, trial)

            class MetricsHistoryCallback(Callback):
                """
                Captures the reported metrics after every validation epoch.
                """

                def __init__(self):
                    super().__init__()

                    self.metric_history = []

                def on_validation_end(self, pl_trainer, pl_module):
                    self.metric_history.append(pl_trainer.callback_metrics)

            metrics_tracker = MetricsHistoryCallback()

            config = load_config(Path(tmp_config_file.name))
            container = create_container(config)

            module = container.module()
            trainer_builder = container.trainer()
            trainer_builder.add_callback(metrics_tracker)

            trainer = trainer_builder.build()
            log_dir = Path(determine_log_dir(trainer))

            # save config of current run to its log dir
            save_config(config, log_dir)

            trainer.fit(
                module,
                train_dataloader=container.train_dataloader(),
                val_dataloaders=container.validation_dataloader()
            )

            save_finished_flag(log_dir)

            def _find_best_value(key: str, best: Callable[[List[float]], float] = min) -> float:
                values = [history_entry[key] for history_entry in metrics_tracker.metric_history]
                return best(values)

            return _find_best_value(objective_metric, objective_best)

    study = optuna.create_study(study_name=study_name, storage=study_storage, load_if_exists=True,
                                direction=study_direction)
    study.optimize(objective, n_trials=num_trails)


@app.command()
def predict(config_file: str = typer.Argument(..., help='the path to the config file'),
            checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file'),
            output_file: Path = typer.Argument(..., help='path where output is written'),
            num_predictions: int = typer.Option(default=20, help='number of predictions to export'),
            gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
            selected_items_file: Optional[Path] = typer.Option(default=None,
                                                               help='only use the item ids for prediction'),
            overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
            log_input: Optional[bool] = typer.Option(default=True, help='enable input logging.')
            ):
    """

    writes the predictions of model (restored from a checkpoint file) to a output file

    :param config_file: the config file used while training the model
    :param checkpoint_file: the checkpoint file of the model
    :param output_file: the path to write the output to
    :param num_predictions: number of predictions
    :param gpu: the number of gpus to use
    :param selected_items_file: the item that should only be considered
    :param overwrite: override the output file
    :param log_input: write the input sequence also to the file
    """
    # checking if the file already exists
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

    test_loader = container.test_dataloader()
    trainer_builder = container.trainer()
    trainer_builder = trainer_builder.set("gpus", gpu)
    trainer = trainer_builder.build()

    def _noop_filter(sample_predictions: np.ndarray):
        return sample_predictions

    filter_predictions = _noop_filter
    selected_items = None

    if selected_items_file is not None:
        selected_items = load_file_with_item_ids(selected_items_file)

        def _selected_items_filter(sample_predictions: np.ndarray):
            return sample_predictions[selected_items]
        filter_predictions = _selected_items_filter

    # open the file and build the writer
    with open(output_file, 'w') as result_file:
        output_writer = build_prediction_writer(result_file, log_input)

        # XXX: currently the predict method returns all batches at once, this is not RAM efficient
        # so we loop through the loader and use only one batch to call the predict method of pytorch lightning
        # replace as soon as this is fixed in pytorch lighting
        class FixedBatchSampler(Sampler):

            def __init__(self, batch_start, batch_size):
                super().__init__(None)
                self.batch_start = batch_start
                self.batch_size = batch_size

            def __iter__(self) -> Iterator[T_co]:
                return iter([range(self.batch_start, self.batch_start + self.batch_size)])

            def __len__(self):
                return 1

        item_tokenizer = container.tokenizer('item')

        for index, batch in enumerate(test_loader):
            sequences = batch[ITEM_SEQ_ENTRY_NAME]
            batch_size = sequences.size()[0]
            batch_start = index * batch_size

            batch_loader = DataLoader(test_loader.dataset, batch_sampler=FixedBatchSampler(batch_start, batch_size),
                                      collate_fn=test_loader.collate_fn)
            prediction_results = trainer.predict(module, dataloaders=batch_loader)

            predictions = prediction_results[0]

            sample_ids = batch[SAMPLE_IDS]
            sequence_position_ids = None
            if 'pos' in batch:
                sequence_position_ids = batch['pos']
            targets = batch[TARGET_ENTRY_NAME]

            def _softmax(array: np.array) -> np.array:
                return np.exp(array) / sum(np.exp(array))

            def _generate_sample_id(sample_ids, sequence_position_ids, sample_index) -> str:
                sample_id = sample_ids[sample_index].item()
                if sequence_position_ids is None:
                    return sample_id

                return f'{sample_id}_{sequence_position_ids[sample_index].item()}'

            for i in range(predictions.shape[0]):
                prediction = filter_predictions(predictions[i])

                scores = _softmax(prediction)

                item_indices = scores[::-1].argsort()[:num_predictions]

                item_ids = item_indices.tolist()

                # when we only want the predictions of selected items
                # the indices are not the item ids anymore, so we have to update them here
                if selected_items is not None:
                    selected_item_ids = [selected_items[i] for i in item_ids]
                    item_ids = selected_item_ids

                tokens = item_tokenizer.convert_ids_to_tokens(item_ids)
                scores[::-1].sort()
                scores = scores.tolist()[:num_predictions]

                sample_id = _generate_sample_id(sample_ids, sequence_position_ids, i)
                true_target = targets[i].item()
                true_target = item_tokenizer.convert_ids_to_tokens(true_target)
                sequence = None
                if log_input:
                    sequence = sequences[i].tolist()

                    # remove padding tokens
                    # TODO: move method
                    def _remove_special_tokens(sequence: List[int], tokenizer: Tokenizer) -> List[int]:
                        for special_token_id in tokenizer.get_special_token_ids():
                            sequence = list(filter(special_token_id.__ne__, sequence))
                        return sequence

                    sequence = _remove_special_tokens(sequence, item_tokenizer)
                    sequence = item_tokenizer.convert_ids_to_tokens(sequence)

                output_writer.write_values(f'{sample_id}', tokens, scores, true_target, sequence)


@app.command()
def evaluate(config_file: str = typer.Argument(..., help='the path to the config file'),
             checkpoint_file: str = typer.Argument(..., help='path to the checkpoint file'),
             output_file: Optional[Path] = typer.Option(default=None, help='path where output is written'),
             gpu: Optional[int] = typer.Option(default=0, help='number of gpus to use.'),
             overwrite: Optional[bool] = typer.Option(default=False, help='overwrite output file if it exists.'),
             seed: Optional[int] = typer.Option(default=None, help='seed used eg for the sampled evaluation')
             ):
    write_results_to_file = output_file is not None
    if write_results_to_file and not overwrite and output_file.exists():
        print(f"${output_file} already exists. If you want to overwrite it, use `--overwrite`.")
        exit(-1)

    if seed is not None:
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
    eval_results = trainer.test(module, test_dataloaders=test_loader, verbose=not write_results_to_file)

    if write_results_to_file:
        with open(output_file, 'w') as output_file_handle:
            result_writer = build_result_writer(output_file_handle)
            # FIXME: currently we have for every model a corresponding module
            # get the first eval results, only one test_dataloader was provided
            result_writer.write_overall_results(type(module).__name__, eval_results[0])


@app.command()
def resume(log_dir: str = typer.Argument(..., help='the path to the logging directory of the run to be resumed'),
           checkpoint_file: Optional[str] = typer.Option(default=None, help="the name of the checkpoint file to resume from")):
    log_dir = Path(log_dir)

    # check for finished flag
    if finished_flag_exists(log_dir):
        print(f"Found a finished flag in '{log_dir}'. Training has finished and will not be resumed.")
        exit(-1)

    # check for config file
    config_file = log_dir / ioutils.PROCESSED_CONFIG_NAME
    if not os.path.isfile(config_file):
        print(f"Could not find '{ioutils.PROCESSED_CONFIG_NAME} in path {log_dir}.")
        exit(-1)

    raw_config = load_config(Path(config_file))
    # determine checkpoint dir:
    checkpoint_dir = Path(
        raw_config.get_config(["trainer", "checkpoint"]).get_or_default("dirpath", log_dir / "checkpoints"))
    # if no checkpoint file is provided we use the last checkpoint.
    if checkpoint_file is None:
        checkpoint_file = "last.ckpt"

    checkpoint_path = checkpoint_dir / checkpoint_file
    if not os.path.isfile(checkpoint_path):
        print("Could not determine the last checkpoint. "
              "You can specify a particular checkpoint via the --checkpoint-file option.")
        exit(-1)

    container = create_container(raw_config)
    module = container.module()

    trainer_builder = container.trainer()
    trainer = trainer_builder.from_checkpoint(checkpoint_path).build()

    train_loader = container.train_dataloader()
    validation_loader = container.validation_dataloader()

    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=validation_loader)

    # Save finished flag
    save_finished_flag(log_dir)


if __name__ == "__main__":
    app()
