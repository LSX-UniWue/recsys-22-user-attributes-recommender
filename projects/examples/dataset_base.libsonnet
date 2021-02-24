{
  example_dataset(base_path, train_dataset_type, batch_size=2, max_sequence_length=5, train_processors=[], validation_processors=[], test_processors=[])::
    local parser = {
         item_column_name: "item_id"
     };
    local loader = {
        batch_size: batch_size,
        max_seq_length: max_sequence_length
    };
    {
        train: {
            loader: {
                dataset: {
                    type: train_dataset_type,
                    csv_file: base_path + "train.csv",
                    csv_file_index: base_path + "train.idx",
                    parser: parser,
                    nip_index_file: base_path + "train.nip.idx",
                    processors: [
                        {
                            type: "tokenizer"
                        }
                    ] + train_processors
                 }
            } + loader,
        },
        validation: {
           loader: {
               dataset: {
                   type: "nextit",
                   csv_file: base_path + "/train.csv",
                   csv_file_index: base_path + "/train.idx",
                   parser: parser,
                   nip_index_file: base_path + "/train.nip.idx",
                   processors: [
                       {
                           type: "tokenizer"
                       }
                   ] + validation_processors
               }
           } + loader
       },
       test: {
           loader: {
               dataset: {
                   type: "nextit",
                   csv_file: base_path + "train.csv",
                   csv_file_index: base_path + "train.idx",
                   parser: parser,
                   nip_index_file: base_path + "train.nip.idx",
                   processors: [
                       {
                           type: "tokenizer"
                       }
                   ] + test_processors
                }
           } + loader,
       }
    },
}
{
    tokenizer(base_path):: {
        item: {
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    delimiter: "\t",
                    file: base_path + "vocab.txt"
                }
            }
        }
    },
}