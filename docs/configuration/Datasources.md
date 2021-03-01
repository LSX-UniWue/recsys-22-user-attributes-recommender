# Datasources


## Parser Config

## Common Constructs

Here we list common data sources configurations.

### Positional Datasource

TODO

### Positive Negative Datasource

This datasource returns the session excluding the last item as sequence (key: `TOOD`) together with
the successor for each sequence step (positive example; key: `TODO`), and a negative sampled item from the item
space, that does not occur in the session or is the successor.


```
{
    type: 'session',
    csv_file: '../tests/example_dataset/train.csv',
    csv_file_index: '../tests/example_dataset/train.idx',
    parser: {
        'item_column_name': 'column_name'
    },
    processors: [
        {
            type: 'tokenizer'
        },
        {
            type: 'pos_neg',
            'seed': 42
        }
    ]
}


```


## Templates for Specific Models


### Positive Negative DataSources Template

This template adds data sources for

- train (Positive Negative Datasource)
- test (Positional Datasource)
- validation (Positional Datasource)

The template is for models that use the complete sequence and train to predict the successor for each
sequence step and compare the scores for the successor with a negative
sample.

It can be triggered by adding the following element instead of `data_sources`:

```
...
pos_neg_data_sources: {
    parser: {
        item_column_name: "column_name"
    },
    batch_size: 64,
    max_seq_length: 200,
    path: "/path",
    train_file_prefix: "train"
    validation_file_prefix: "train",
    test_file_prefix: "train",
    seed: 42
},
...
```

By default, the template configures the framework to 

The following config parameters are available:

- `parser`: configs the parser for the csv file
- `batch_size`: the batch size to use, if you want to override this
  for train, test or validation, add a `_batch_size` element to the
  element
- `seed`: the seed used to generate negative samples
