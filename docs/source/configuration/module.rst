.. _config module:
Module
======

Metrics
-------

The following metrics are implemented in this framework with the keys to
use for configuration:

-  Recall / HR (``recall``)
-  Precision (``precision``)
-  F1 (``F1``)
-  DCG (``DCG``)
-  NDCG (``NDCG``)
-  MRR (``MRR``)

For each metric you can provide one or more different ``k``\ s to
evaluate the Metric@k value. The metrics can be access (e.g. in the
checkpoint), via ``KEY@k``.

Metric Evaluation
-----------------

There are three evaluation strategies available in the framework:

-  ``full``: the metrics are evaluated on the complete item space
-  ``sampled``: the metrics are evaluated on the positive item(s) and
   ``s`` sampled negative items (given a probability)
-  ``fixed``: the metrics are evaluated on a fixed subset of the item
   space

Full
~~~~

If you want to validate your model or evaluate your model how good your
model ranks all items of the item space, you can specify a metrics
section under module. For each metric you can specify which ``k``\ s
should be evaluated.

.. code:: json

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

Sampled Metrics
~~~~~~~~~~~~~~~

In contrast to metrics the sampled metrics configuration only samples
items from the item space to evaluate it with target item(s).

.. code:: json

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

The ``sampled`` metrics config, requires the following parameters:

-  ``sample_probability_file``: The configurable file contains in the
   i-th line the probability of the (i-1) item based on the vocabulary
   files.
-  ``num_negative_samples``: The number of negative samples to draw from
   the provided probability file.
-  ``metrics`` you can define all metrics you can also define using all
   items of the dataset.

TODO: point to the help script to calculate the file

Fixed Subset
~~~~~~~~~~~~

This metric only evaluates a fixed set of items.

.. code:: json

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

The ``fixed`` metrics config, requires the following parameters:

-  ``item_file``: The configurable file contains the item ids of the
   subset to evaluate (item id line by line).
-  ``metrics`` you can define all metrics you can also define using all
   items of the dataset.

Module/Model Configuration
----

You can configure your module/model by changing the parameters passed to its ``__init__``
function via the corresponding configuration sections. If your model requires configuration aside from integral types such as ``int``, ``str``, etc. you can specify
annotate the model parameter in the ``__init__`` function with one of the following annotations:

- ``InjectTokenizer(feature_name)``: ASME will inject a Tokenizer instance for the given ``feature_name`` into this parameter.
- ``InjectVocabularySize(feature_name)``: ASME will inject the size of the vocabulary for the given ``feature_name`` into this parameter.
- ``InjectModel(model_cls, config_section_path)``: ASME will instantiate an instance of ``model_cls`` via the ``GenericModuleFactory`` using the configuration
  section specified by ``config_section_path`` to fill its parameters. If ``config_section_path`` is not provided, ASME assumes that the corresponding
  configuration section has the same name as the parameter. Note that ``config_section_path`` is interpreted relative to the module/model configuration.
- ``InjectClass(config_section_path)``: ASME will instantiate an arbitrary class to fill this parameter. The configuration section ``config_section_path`` has to
  contain the following fields:
    - ``cls_name``: The name of the class that should be instantiated.
    - ``module_name``: The fully qualified name of the module that contains the class referenced by ``cls_name``.
    - ``parameters``: This is an optional list of parameters to be passed to the class upon instantiation. If any parameter is a dictionary that contains the key
      ``cls_name``, the corresponding object is recursively instantiated.
- ``InjectList(config_section_path)``: ASME expects the configuration section ``config_section_path`` to be a list. Each element of this list is recursively built
  if necessary and the resulting list is injected into the parameter.
- ``InjectDict(config_section_path)``: ASME expects the configuration section ``config_section_path`` to be a dictionary. Each element of this dictionary is recursively built
  if necessary and the resulting dictionary is injected into the parameter.