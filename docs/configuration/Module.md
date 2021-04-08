# Module

## Metrics

The following asme.metrics are implemented in this framework with the keys to use for configuration:

- Recall / HR (`recall`)
- Precision (`precision`)
- F1 (`F1`)
- DCG (`DCG`)
- NDCG (`NDCG`)
- MRR (`MRR`)

For each metric you can provide one or more different `k`s to evaluate the Metric@k value.
The asme.metrics can be access (e.g. in the checkpoint), via `KEY@k`.

## Metric Evaluation

There are three evaluation strategies available in the framework:

- `full`: the asme.metrics are evaluated on the complete item space
- `sampled`: the asme.metrics are evaluated on the positive item(s) and `s` sampled negative items (given a probability)
- `fixed`: the asme.metrics are evaluated on a fixed subset of the item space

### Full

If you want to validate your model or evaluate your model how good your model ranks all items
of the item space, you can specify a asme.metrics section under module.
For each metric you can specify which `k`s should be evaluated.

``` json
...
module: {
        ...
        asme.metrics: {
            full: {
                asme.metrics: {
                    recall: [1, 3, 5]
                }
            },
            ...
        },
        ...
}
```


### Sampled Metrics

In contrast to asme.metrics the sampled asme.metrics configuration only samples items from the item space
to evaluate it with target item(s).

``` json
...
module: {
        ...
        asme.metrics: {
            sampled: {
                sample_probability_file: PATH_TO_FILE,
                num_negative_samples: 100,
                asme.metrics: {
                    recall: [1, 3, 5]
                }
            },
            ...
        },
        ...
}
```

The `sampled` asme.metrics config, requires the following parameters:

- `sample_probability_file`: The configurable file contains in the i-th line the probability of the (i-1) item based
on the vocabulary files.
- `num_negative_samples`: The number of negative samples to draw from the provided probability file.
- `asme.metrics` you can define all asme.metrics you can also define using all items of the dataset.

TODO: point to the help script to calculate the file


### Fixed Subset

This metric only evaluates a fixed set of items.

``` json
...
module: {
        ...
        asme.metrics: {
            fixed: {
                item_file: PATH_TO_FILE,
                asme.metrics: {
                    recall: [1, 3, 5]
                }
            },
            ...
        },
        ...
}
```

The `fixed` asme.metrics config, requires the following parameters:

- `item_file`: The configurable file contains the item ids of the subset to evaluate (item id line by line).
- `asme.metrics` you can define all asme.metrics you can also define using all items of the dataset.
