from models.bert4rec.bert4rec_model import BERT4RecModel
from modules import BERT4RecModule

from runner.sasrec import build_standard_trainer, build_tokenizer_provider, \
    build_session_loader_provider_factory, build_nextitem_loader_provider_factory

from dependency_injector import containers, providers


class BERT4RecContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # tokenizer
    tokenizer = build_tokenizer_provider(config)

    # model
    model = providers.Singleton(
        BERT4RecModel,
        config.model.transformer_hidden_size,
        config.model.num_transformer_heads,
        config.model.num_transformer_layers,
        config.model.item_vocab_size,
        config.model.max_seq_length,
        config.model.dropout
    )

    module = providers.Singleton(
        BERT4RecModule,
        model,
        config.module.batch_size,
        config.module.mask_probability,
        config.module.learning_rate,
        config.module.beta_1,
        config.module.beta_2,
        config.module.weight_decay,
        config.module.num_warmup_steps,
        tokenizer,
        config.module.batch_first,
        config.module.metrics_k
    )

    # loaders
    train_loader = build_session_loader_provider_factory(config, tokenizer, lambda config: config.datasets.train)
    validation_loader = build_nextitem_loader_provider_factory(config, tokenizer, lambda config: config.datasets.validation)
    test_loader = build_nextitem_loader_provider_factory(config, tokenizer, lambda config: config.datasets.test)

    # trainer
    trainer = build_standard_trainer(config)


def main():
    container = BERT4RecContainer()
    container.config.from_yaml("util/bert4rec_config.yaml")
    module = container.module()

    trainer = container.trainer()
    trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())


if __name__ == "__main__":
    main()
