local base_path = "../dataset/ml-1m_5_4_0/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};
{
    templates: {

        par_pos_neg_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 4,
                max_seq_length: max_seq_length,
                num_workers: 4

            },
            path: base_path,
            file_prefix: "ml-1m",
            split_type: "leave_one_out",
            seed: 123456
        }
    },
    module: {
        type: "cosrec",
        learning_rate: 0.001,
        weight_decay: 0.01,
        metrics: {
            full: {
                metrics: metrics
            },

        },
        model: {
            user_vocab_size: 0,
            max_seq_length: 5,
            embed_dim: 50,
            block_num: 2,
            block_dim: [128, 256],
            fc_dim: 150,
            activation_function: 'relu',
            dropout: 0.5
        }
    },
    tokenizers: {
        item: {
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: base_path + "vocab_title.txt"
                }
            }
        }
    },
    trainer: {

        checkpoint: {
            monitor: "recall@10",
            save_top_k: 3,
            mode: 'max'
        },
        max_epochs: 800,
        check_val_every_n_epoch: 100
    }
}