from typing import Any, Dict

from dependency_injector import containers, providers

from models.bert4rec.bert4rec_model import BERT4RecModel
from models.bert4rec.bert4rec_model_2 import BERT4RecModel2
from models.caser.caser_model import CaserModel
from models.gru.gru_model import GRUSeqItemRecommenderModel
from models.narm.narm_model import NarmModel
from models.sasrec.sas_rec_model import SASRecModel
from modules import BERT4RecModule, CaserModule, SASRecModule
from modules.gru_module import GRUModule
from modules.narm_module import NarmModule
from runner.util.provider_utils import build_tokenizer_provider, build_session_loader_provider_factory, \
    build_nextitem_loader_provider_factory, build_posneg_loader_provider_factory, build_standard_trainer, \
    build_metrics_provider, build_processors_provider

DEFAULT_PROCESSORS = {
    'tokenizer_processor': {
        'tokenize': True
    }
}

CONFIG_KEY_MODULE = 'module'

MODULE_ADAM_OPTIMIZER_DEFAULT_VALUES = {
    CONFIG_KEY_MODULE: {
        'learning_rate': 0.001,
        'beta_1': 0.99,
        'beta_2': 0.998,
    }
}


def build_default_config() -> providers.Configuration:
    config = providers.Configuration()
    # init some default values
    config.from_dict({
        'trainer': {
            'experiment_name': 'basic_experiment',
            'limit_train_batches': 1.0,
            'limit_val_batches': 1.0,
            'gradient_clip_val': 0.0,
            'default_root_dir': '/tmp/',
            'max_epochs': 20
        },
        'model': {
            'embedding_mode': None
        },
        'datasets': {
            'train': _build_default_dataset_config(shuffle=True),
            'validation': _build_default_dataset_config(shuffle=False),
            'test': _build_default_dataset_config(shuffle=False)
        }
    })
    return config


def _build_default_dataset_config(shuffle: bool,
                                  processors: Dict[str, Dict[str, Any]] = None
                                  ) -> Dict[str, Any]:
    if processors is None:
        processors = DEFAULT_PROCESSORS

    return {
        'dataset': {
            'parser': {
                'additional_features': {},
                'item_separator': None,
                'delimiter': '\t'
            },
            'processors': processors
        },
        'loader': {
            'num_workers': 0,
            'shuffle': shuffle,
            'max_seq_step_length': None
        }
    }


def _build_pos_neg_model_processors() -> Dict[str, Any]:
    processors = DEFAULT_PROCESSORS.copy()
    processors['pos_neg_sampler'] = {
        'pos_neg_sampling': True
    }
    return {
        'datasets': {
            'train': _build_default_dataset_config(shuffle=True, processors=processors)
        }
    }


# XXX: find a better way to allow
def _kwargs_adapter(clazz, kwargs):
    return clazz(**kwargs)


class BERT4RecContainer(containers.DeclarativeContainer):

    config = build_default_config()
    # some model specific default value
    config.from_dict({
        CONFIG_KEY_MODULE: {
            'num_warmup_steps': 10000,
            'batch_first': True,
            'mask_probability': 0.2,
            'weight_decay': 0.01
        }
    })
    config.from_dict(MODULE_ADAM_OPTIMIZER_DEFAULT_VALUES)

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    # model
    model = providers.Singleton(_kwargs_adapter, BERT4RecModel, config.model)

    module_config = config.module

    module = providers.Singleton(
        BERT4RecModule,
        model=model,
        mask_probability=module_config.mask_probability,
        learning_rate=module_config.learning_rate,
        beta_1=module_config.beta_1,
        beta_2=module_config.beta_2,
        weight_decay=module_config.weight_decay,
        num_warmup_steps=module_config.num_warmup_steps,
        tokenizer=tokenizer,
        batch_first=module_config.batch_first,
    )

    # dataset config
    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    processors_objects = {'tokenizer': tokenizer}
    train_processors = build_processors_provider(train_dataset_config.dataset.processors, processors_objects)
    validation_processors = build_processors_provider(validation_dataset_config.dataset.processors, processors_objects)
    test_processors = build_processors_provider(test_dataset_config.dataset.processors, processors_objects)

    # loaders
    train_loader = build_session_loader_provider_factory(train_dataset_config, tokenizer, train_processors)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer,
                                                               validation_processors)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer, test_processors)

    # trainer
    trainer = build_standard_trainer(config)


class CaserContainer(containers.DeclarativeContainer):

    config = build_default_config()
    # add pos neg sampler by default TODO: can we do this later?
    config.from_dict(_build_pos_neg_model_processors())

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    # model
    model = providers.Singleton(_kwargs_adapter, CaserModel, config.model)

    module_config = config.module


    module = providers.Singleton(
        CaserModule,
        model=model,
        tokenizer=tokenizer,
        learning_rate=module_config.learning_rate,
        weight_decay=module_config.weight_decay,
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    processors_objects = {'tokenizer': tokenizer}
    train_processors = build_processors_provider(train_dataset_config.dataset.processors, processors_objects)
    validation_processors = build_processors_provider(validation_dataset_config.dataset.processors, processors_objects)
    test_processors = build_processors_provider(test_dataset_config.dataset.processors, processors_objects)

    # loaders
    train_loader = build_posneg_loader_provider_factory(train_dataset_config, tokenizer, train_processors)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer,
                                                               validation_processors)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer, test_processors)

    # trainer
    trainer = build_standard_trainer(config)


class SASRecContainer(containers.DeclarativeContainer):

    config = build_default_config()
    # add pos neg sampler by default
    config.from_dict(_build_pos_neg_model_processors())

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    model_config = config.model

    # model
    model = providers.Singleton(_kwargs_adapter, SASRecModel, config.model)

    module_config = config.module

    module = providers.Singleton(
        SASRecModule,
        model=model,
        learning_rate=module_config.learning_rate,
        beta_1=module_config.beta_1,
        beta_2=module_config.beta_2,
        tokenizer=tokenizer,
        batch_first=module_config.batch_first,
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    processors_objects = {'tokenizer': tokenizer}
    train_processors = build_processors_provider(train_dataset_config.dataset.processors, processors_objects)
    validation_processors = build_processors_provider(validation_dataset_config.dataset.processors, processors_objects)
    test_processors = build_processors_provider(test_dataset_config.dataset.processors, processors_objects)

    # loaders
    train_loader = build_posneg_loader_provider_factory(train_dataset_config, tokenizer, train_processors)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer,
                                                               validation_processors)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer, test_processors)

    trainer = build_standard_trainer(config)


class NarmContainer(containers.DeclarativeContainer):

    config = build_default_config()

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    # model
    model = providers.Singleton(_kwargs_adapter, NarmModel, config.model)

    module_config = config.module

    module = providers.Singleton(
        NarmModule,
        model,
        module_config.batch_size,
        module_config.learning_rate,
        module_config.beta_1,
        module_config.beta_2,
        tokenizer,
        module_config.batch_first,
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    processors_objects = {'tokenizer': tokenizer}
    train_processors = build_processors_provider(train_dataset_config.dataset.processors, processors_objects)
    validation_processors = build_processors_provider(validation_dataset_config.dataset.processors, processors_objects)
    test_processors = build_processors_provider(test_dataset_config.dataset.processors, processors_objects)

    # loaders
    train_loader = build_nextitem_loader_provider_factory(train_dataset_config, tokenizer, train_processors)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer,
                                                               validation_processors)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer, test_processors)

    trainer = build_standard_trainer(config)


class GRUContainer(containers.DeclarativeContainer):

    config = build_default_config()
    config.from_dict(MODULE_ADAM_OPTIMIZER_DEFAULT_VALUES)

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    model_config = config.model

    # model
    model = providers.Singleton(_kwargs_adapter, GRUSeqItemRecommenderModel, config.model)

    module_config = config.module

    module = providers.Singleton(
        GRUModule,
        model,
        module_config.learning_rate,
        module_config.beta_1,
        module_config.beta_2,
        tokenizer,
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    processors_objects = {'tokenizer': tokenizer}
    train_processors = build_processors_provider(train_dataset_config.dataset.processors, processors_objects)
    validation_processors = build_processors_provider(validation_dataset_config.dataset.processors, processors_objects)
    test_processors = build_processors_provider(test_dataset_config.dataset.processors, processors_objects)

    # loaders
    train_loader = build_nextitem_loader_provider_factory(train_dataset_config, tokenizer, train_processors)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer,
                                                               validation_processors)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer, test_processors)

    trainer = build_standard_trainer(config)
