# Code
The folder contains the following elements:
 * 📜 [Data Utils](DataUtils.py) for loading videos and annotations
 * 📂 Models
   * 📜 [Mamba implementation](Models/Mamba.py) by [Vedant Jumle](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-tf.keras-and-tensorflow-32d6d4b32546), edited by Dominik Beese
   * 📜 [Transformer implementation](Models/Transformer.py) by [The TensorFlow Authors](https://github.com/tensorflow/models/blob/v2.15.0/official/nlp/modeling/layers/transformer_encoder_block.py#L24-L412), edited by Dominik Beese
   * 📜 [CNN model wrapper](Models/CNN.py)
   * 📜 [RNN model implementation](Models/RNN.py)
   * 📜 [CNN-RNN architecture implementation](Models/CNN_RNN.py)
   * 📜 [CNN model with fine-tuning head](Models/CNNForFineTuning.py)
 * 📂 Training
   * 📜 [Grid search implementation](Training/HyperparameterTuner.py)
   * 📜 Additional tensorflow [Callbacks](Training/Callbacks.py)
   * 📜 Additional tensorflow [Losses](Training/Losses.py)
   * 📜 Additional tensorflow [Metrics](Training/Metrics.py)

The code is tested using Python 3.10.6. The required libraries are listed in the [`requirements.txt`](requirements.txt).


## Documentation for the Grid search implementation
First, the hyperparameter tuner needs to be initialized:
```python
tuner = HyperparameterTuner.GridTuner(...)
```

Options for `HyperparameterTuner.GridTuner` are:
| Parameter | Type | Description |
|-----------|------|-------------|
| `build_model` | callable | the function that builds and returns a model |
| `objective` | str | the objective to minimize |
| `direction` | str | whether to minimize or maximize the objective |
| `executions_per_trial` | int | the number of executions, i.e. model trainings, to do for each hyperparameter configuration |
| `output_dir` | str | the directory to save logs, predictions and model weights to |
| `output_predictions` | bool | whether to compute and save predictions |
| `save_best_model_weights` | bool | whether to save or discard the weights of the best performing model |

Next, the grid search can be started:
```python
progress = tuner.search(...)
```

Options for `HyperparameterTuner.GridTuner.search()` are:
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | tf.Dataset | training data |
| `start_with_trial` | int | skip certain trials from the grid search |
| `end_with_trial` | int | skip certain trials from the grid search |
| `validate_execution` | callable | function to run before each execution that returns true if the (already finished) execution is valid or not, it gets the history and the current hyperparameters |
| `retry_execution` | callable | function to run after each execution that returns true if the current execution should be discarded and repeated, it gets the model, the history and the current hyperparameters |
| `run_after_each_execution` | callable | function that runs after each execution, it gets the model, the history and the current hyperparameters |
| `seed` | int | if None, use random seeds, otherwise uses the given seed<br>_Note:_ if you use, `executions_per_trial` > 1, it does not make sense to fix the seed |
| `tensorboard` | bool | whether to log using tensorboard |
| `**kwargs` | — | for `model.fit()` |
