from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.bert4rec.bert4rec_model import BERT4RecModel
from modules import BERT4RecModule

from runner.sasrec import provide_vocabulary, provide_tokenizer, provide_nextitem_dataset, provide_posneg_dataset, \
    provide_posneg_loader, provide_nextit_loader
from tokenization.vocabulary import CSVVocabularyReaderWriter

from dependency_injector import containers, providers


class BERT4RecContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # tokenizer
    vocabulary_serializer = providers.Singleton(CSVVocabularyReaderWriter, config.tokenizer.vocabulary.delimiter)
    vocabulary = providers.Singleton(provide_vocabulary, vocabulary_serializer, config.tokenizer.vocabulary.file)
    tokenizer = providers.Singleton(provide_tokenizer, vocabulary, config.tokenizer.special_tokens)

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
        config.module.learning_rate,
        config.module.beta_1,
        config.module.beta_2,
        config.module.weight_decay,
        config.module.num_warmup_steps,
        tokenizer,
        config.module.batch_first,
        config.module.metrics_k
    )

    train_dataset = providers.Factory(
        provide_posneg_dataset,
        config.datasets.train.dataset.csv_file,
        config.datasets.train.dataset.csv_file_index,
        tokenizer,
        config.datasets.train.dataset.delimiter,
        config.datasets.train.dataset.item_column_name
    )

    validation_dataset = providers.Factory(
        provide_nextitem_dataset,
        config.datasets.validation.dataset.csv_file,
        config.datasets.validation.dataset.csv_file_index,
        config.datasets.validation.dataset.nip_index_file,
        tokenizer,
        config.datasets.validation.dataset.delimiter,
        config.datasets.validation.dataset.item_column_name
    )
    
    test_dataset = providers.Factory(
        provide_nextitem_dataset,
        config.datasets.test.dataset.csv_file,
        config.datasets.test.dataset.csv_file_index,
        config.datasets.test.dataset.nip_index_file,
        tokenizer,
        config.datasets.test.dataset.delimiter,
        config.datasets.test.dataset.item_column_name
    )
    
    train_loader = providers.Factory(
        provide_posneg_loader,
        train_dataset,
        config.datasets.train.loader.batch_size,
        config.datasets.train.loader.max_seq_length,
        tokenizer
    )

    validation_loader = providers.Factory(
        provide_nextit_loader,
        validation_dataset,
        config.datasets.validation.loader.batch_size,
        config.datasets.validation.loader.max_seq_length,
        tokenizer
    )

    test_loader = providers.Factory(
        provide_nextit_loader,
        test_dataset,
        config.datasets.test.loader.batch_size,
        config.datasets.test.loader.max_seq_length,
        tokenizer
    )

    checkpoint = providers.Singleton(
        ModelCheckpoint,
        filepath=config.trainer.checkpoint.filepath,
        monitor=config.trainer.checkpoint.monitor,
        save_top_k=config.trainer.checkpoint.save_top_k,

    )

    trainer = providers.Singleton(
        Trainer,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        default_root_dir=config.trainer.default_root_dir,
        checkpoint_callback=checkpoint,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gpus=config.trainer.gpus
    )


def main():
    container = BERT4RecContainer()
    container.config.from_yaml("util/bert4rec_config.yaml")
    module = container.module()

    trainer = container.trainer()
    trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())


if __name__ == "__main__":
    main()


