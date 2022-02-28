User Guide
==========

Pre-Requisites:

- Built the virtual environment as described in the :ref:`Project Overview <_project_overview>`

First of all you need to activate your virtual environment. This is
accomplished using one of the two following commands

.. code:: bash

    # Either
    poetry shell
    # or
    source recommender/venv/recommender/bin/activate

The components of this framework can be executed using the `Runner <../asme/runner>`__.

Using integrated Datasets
------------------------
ASME includes several datasets such as ml-1m and ml-20m that can be preprocessed in order to fit your needs.
The process of selecting a dataset and specifying its preprocessing is facilitated via corresponding entries in the configuration file.
The dataset can be choosen by setting the `name` parameter in the `datamodule` section of the configuration to the matching name.
Preprocessing can than be adapted to fit your needs by setting appropriate parameters in the `preprocessing` section of the `datamodule` configuration entry.


Generally, ASME will generate three types of splits for each dataset:

- Ratio split: The data is split by session into three sets for training, validation and testing.
- Leave-One-Out (LOO) split: The data is split intra-session, i.e. a session of n items is decomposed into a n - 2 item training session, a 1 item validation and a 1 item test session.
- Leave-Percentage-Out (LPO) split: Similar to the LOO split, the data is partioned intra-session. Specifically, the first t% of each session is used for training, the next v% for validation and the final (1 - t - v)% for testing.


You can customize the preprocessing process by setting the following parameters:


- Ratio split:
    - ratio_split_min_item_feedback:        The minimum number of interactions necessary for an item to be kept in the dataset.
    - ratio_split_min_sequence_length:      The minimum number of interactions in a session for it to be kept in the dataset.
    - ratio_split_train_percentage:         The fraction of session to (approximately) include into the training set.
    - ratio_split_validation_percentage:    The fraction of session to (approximately) include into the validation set.
    - ratio_split_test_percentage:          The fraction of session to (approximately) include into the test set.
    - ratio_split_window_markov_length:     The size of the sliding window that is used to extract samples from each session.
    - ratio_split_window_target_length:     The size of the sliding window that is used to extract targets from each session.
    - ratio_split_session_end_offset:       The distance between the session end and the last position for the sliding window.

- Leave-One-Out split:
    - loo_split_min_item_feedback:      The minimum number of interactions necessary for an item to be kept in the dataset.
    - loo_split_min_sequence_length:    The minimum number of interactions in a session for it to be kept in the dataset.

- Leave-Percentage-Out split:
    - lpo_split_min_item_feedback:      The minimum number of interactions necessary for an item to be kept in the dataset.
    - lpo_split_min_sequence_length:    The minimum number of interactions in a session for it to be kept in the dataset.
    - lpo_split_train_percentage:       The fraction of session to (approximately) include into the training set.
    - lpo_split_validation_percentage:  The fraction of session to (approximately) include into the validation set.
    - lpo_split_test_percentage:        The fraction of session to (approximately) include into the test set.
    - lpo_split_min_train_length:       The minimum size of each session in the training set.
    - lpo_split_min_validation_length:  The minimum size of each session in the validation set.
    - lpo_split_min_test_length:        The minimum size of each session in the test set.


Training implemented Models
---------------------------

Executing Trained Models
------------------------

