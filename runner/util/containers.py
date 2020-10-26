from dependency_injector import containers, providers

from models.bert4rec.bert4rec_model import BERT4RecModel
from models.caser.caser_model import CaserModel
from models.narm.narm_model import NarmModel
from models.sasrec.sas_rec_model import SASRecModel
from modules import BERT4RecModule, CaserModule, SASRecModule
from modules.narm_module import NarmModule
from runner.util.provider_utils import build_tokenizer_provider, build_session_loader_provider_factory, \
    build_nextitem_loader_provider_factory, build_posneg_loader_provider_factory, build_standard_trainer


class BERT4RecContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    model_config = config.model

    # model
    model = providers.Singleton(
        BERT4RecModel,
        model_config.transformer_hidden_size,
        model_config.num_transformer_heads,
        model_config.num_transformer_layers,
        model_config.item_vocab_size,
        model_config.max_seq_length,
        model_config.dropout
    )

    module_config = config.module

    module = providers.Singleton(
        BERT4RecModule,
        model,
        module_config.mask_probability,
        module_config.learning_rate,
        module_config.beta_1,
        module_config.beta_2,
        module_config.weight_decay,
        module_config.num_warmup_steps,
        tokenizer,
        module_config.batch_first,
        module_config.metrics_k
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    # loaders
    train_loader = build_session_loader_provider_factory(train_dataset_config, tokenizer)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer)

    # trainer
    trainer = build_standard_trainer(config)


class CaserContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    model_config = config.model

    # model
    model = providers.Singleton(
        CaserModel,
        model_config.embedding_size,
        model_config.item_vocab_size,
        model_config.user_vocab_size,
        model_config.max_seq_length,
        model_config.num_vertical_filters,
        model_config.num_horizontal_filters,
        model_config.conv_activation_fn,
        model_config.fc_activation_fn,
        model_config.dropout
    )

    module_config = config.module

    module = providers.Singleton(
        CaserModule,
        model,
        tokenizer,
        module_config.learning_rate,
        module_config.weight_decay,
        module_config.metrics_k
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    # loaders
    train_loader = build_posneg_loader_provider_factory(train_dataset_config, tokenizer)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer)

    # trainer
    trainer = build_standard_trainer(config)


class SASRecContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    # model
    model = providers.Singleton(
        SASRecModel,
        config.model.transformer_hidden_size,
        config.model.num_transformer_heads,
        config.model.num_transformer_layers,
        config.model.item_vocab_size,
        config.model.max_seq_length,
        config.model.dropout
    )

    module = providers.Singleton(
        SASRecModule,
        model,
        config.module.batch_size,
        config.module.learning_rate,
        config.module.beta_1,
        config.module.beta_2,
        tokenizer,
        config.module.batch_first,
        config.module.metrics_k
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    # loaders
    train_loader = build_posneg_loader_provider_factory(train_dataset_config, tokenizer)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer)

    trainer = build_standard_trainer(config)


class NarmContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    # model
    model = providers.Singleton(
        NarmModel,
        config.model.item_vocab_size,
        config.model.item_embedding_size,
        config.model.global_encoder_size,
        config.model.global_encoder_num_layers,
        config.model.embedding_dropout,
        config.model.context_dropout,
        config.model.batch_first
    )

    module = providers.Singleton(
        NarmModule,
        model,
        config.module.batch_size,
        config.module.learning_rate,
        config.module.beta_1,
        config.module.beta_2,
        tokenizer,
        config.module.batch_first,
        config.module.metrics_k
    )

    train_dataset_config = config.datasets.train
    validation_dataset_config = config.datasets.validation
    test_dataset_config = config.datasets.test

    # loaders
    train_loader = build_nextitem_loader_provider_factory(train_dataset_config, tokenizer)
    validation_loader = build_nextitem_loader_provider_factory(validation_dataset_config, tokenizer)
    test_loader = build_nextitem_loader_provider_factory(test_dataset_config, tokenizer)

    trainer = build_standard_trainer(config)