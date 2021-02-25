# Module

## Metrics

The following metrics are implemented in this framework with the keys to use for configuration:

- Recall / HR (`recall`)
- Precision (`precision`)
- F1 (`f1`)
- DCG (`dcg`)
- NDCG (`ndcg`)
- MRR (`mrr`)

For each metric you can provide one or more different `k`s to evaluate the Metric@k value.

## Supported Metrics

### Metrics

If you want to validate your model or evaluate your model how good your model ranks all items
of the item space, you can specify a metrics section under module.
For each metric you can specify which `k`s should be evaluated.

```
...
module: {
        ...
        metrics: {
            full: {
                metrics: {
                    recall: [1, 3, 5]
                }
            },
            ...
        },
        ...
}
```


### Sampled Metrics

In contrast to metrics the sampled metrics configuration only samples items from the item space
to evaluate it with target item(s).

```
...
module: {
        ...
        metrics: {
            sampled: {
                sample_probability_file: PATH_TO_FILE,
                num_negative_samples: 100,
                metrics: {
                    recall: [1, 3, 5]
                }
            },
            ...
        },
        ...
}
```

The configurable file `sample_probability_file` contains in the i-th line the propability of the (i-1) item based
on the vocabulary file.

Under `metrics` you can define all metrics you can also define using all items of the dataset.

TODO: point to the help script to calculate the file

### Fixed Subset

This metric only evaluates a fixed set of items.

```
...
module: {
        ...
        metrics: {
            fixed: {
                item_file: PATH_TO_FILE,
                metrics: {
                    recall: [1, 3, 5]
                }
            },
            ...
        },
        ...
}
```

The 