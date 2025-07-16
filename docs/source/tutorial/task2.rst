Task 2: Train - Test - Model Evaluation
=======================================

Overview
--------

This task involves training, testing, and evaluating model performance
using a specified configuration and dataset.

.. code-block:: bash

    python bin/train.py \
        --config configs/config.yml \
        --original <original_style> \
        --style <destination_style> \
        --checkpoints checkpoints \
        --data data

.. code-block:: bash

    python bin/test.py \
        --config configs/config.yml \
        --original <original_style> \
        --style <destination_style> \
        --checkpoints <best_checkpoint_weight> \
        --data data \
        --results results

.. code-block:: bash

    python bin/eval.py \
        --original <original_style> \
        --style <destination_style> \
        --results <generated_results>