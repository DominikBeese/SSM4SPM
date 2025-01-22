''' Author: Dominik Beese
>>> Trainer for the frame-based experiment.
<<<
'''

import sys
from os.path import join
import json

import sklearn
import numpy as np
import pandas as pd

import tensorflow as tf


### CONFIGURATION ###

gpu_devices = [0] # enter gpus to use

# Frame2-IA (large)
architectures = ['ResNet152']
experiment_name = 'Frame2-IA' # the name of the experiment to train for
output_config = 'd' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [2e-6, 1e-6, 7e-7],
	'head_dimension': [1000],
}
validation_batch_size = 8 #"""

# global parameters
hyperparameters['epochs'] = [200]
inference_batch_size = 8
executions_per_trial = 3

# load configuration
with open('../configuration.json', 'r', encoding='UTF-8') as file:
	config = json.load(file)

# import code and model
sys.path.append(config['code-folder'])
from Models import ModelUtils, CNNForFineTuning
import GPUUtils, DataUtils
from Training import HyperparameterTuner, Losses, Metrics, Callbacks

# setup paths
data_file = join('.', experiment_name, 'Splits', '%s-run%d.json')
data_path = join(config['root-data-folder'], 'Cataract-1K')

# select gpus
GPUUtils.set_visible_devices(gpu_devices)


### EXPERIMENT ###

for architecture in architectures:
	made_progress2 = False
	for split in split_names or [None]:
		made_progress = False
		for run in split_runs:
			print()
			print('='*50)
			print('>>', architecture, 'Split', split, 'Run', run)
			print('='*50)
			
			### Setup ###
			
			# parameters
			image_size = ModelUtils.sizeof(architecture)
			
			# load data
			dataset = {k: pd.read_json(data_file % (k + ('-' + str(split) if split else ''), run)) for k in ['train', 'dev', 'test']}
			encoded_dataset = {k: DataUtils.load_from_splits(dataset[k], data_path, image_size=image_size, model_name=None, quick=True) for k in ['train', 'dev']}
			
			### Model Builder ###
			
			def build_model(hp):
				# get parameters
				hp.get('batch_size', values=hyperparameters['batch_size'])
				hp.get('epochs', values=hyperparameters['epochs'])
				learning_rate = hp.get('learning_rate', values=hyperparameters['learning_rate'])
				head_dimension = hp.get('head_dimension', values=hyperparameters['head_dimension'])
				
				# build model
				output_type = [{'x': 'softmax', 'd': 'sigmoid'}[c] for c in output_config]
				if len(output_config) > 1:
					labels = dataset['train']['label'].apply(pd.Series)
					labels = [labels[i] for i in labels.columns]
				else: labels = [dataset['train']['label']]
				labels = [l.explode().dropna().to_numpy() if c == 'x' else np.where(pd.DataFrame(l.to_list()).to_numpy() == 1)[1] for l, c in zip(labels, output_config)]
				num_labels = [np.max(l)+1 for l in labels]
				class_weight = [sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(l), y=l) for l in labels]
				model = CNNForFineTuning(architecture=architecture, head_dimension=head_dimension, output_type=output_type, num_labels=num_labels, **model_kwargs)
				loss = [{'x': Losses.WeightedSparseCategoricalCrossentropy, 'd': Losses.WeightedBinaryCrossentropy}[c] for c in output_config]
				metric = [{'x': tf.keras.metrics.SparseCategoricalAccuracy, 'd': tf.keras.metrics.BinaryAccuracy}[c] for c in output_config]
				model.compile(
					optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
					loss=[l(class_weight=cw) for l, cw in zip(loss, class_weight)],
					metrics=[metric[0](name='accuracy')] if len(metric) == 1 else list(), # no metric for multi-outputs
				)
				return model
			
			### Train ###
			
			# create grid tuner
			output_dir = join(experiment_name, 'Evaluation-' + architecture.replace('-', '_'), '%srun%d' % (str(split) + '-' if split else '', run))
			tuner = HyperparameterTuner.GridTuner(
				build_model,
				objective='val_loss',
				direction='minimize',
				executions_per_trial=executions_per_trial,
				output_predictions=False, ## True
				save_best_model_weights=True,
				output_dir=output_dir,
			)
			
			# start grid search
			progress = tuner.search(
				data=encoded_dataset['train'],
				validation_data=encoded_dataset['dev'],
				validation_batch_size=validation_batch_size,
				validation_freq=1,
				tensorboard=False,
				callbacks=[
					Callbacks.EarlyStopping(target='val_loss', earliest_stop_epoch=20, patience=10),
					##Callbacks.EarlyFailing(epoch=20, target='accuracy', value=0.2),
				],
				##validate_execution=lambda history, **kwargs: history['metrics']['accuracy'][20 // 2] > 0.2,
				##retry_execution=lambda model, retries, **kwargs: model.stopped_by == 'EarlyFailing' and retries < 10,
			)
			if not progress: continue
			
			### Predict ###
			
			# load model weights
			with open(join(output_dir, 'best-configuration.json'), 'r', encoding='UTF-8') as file: best = json.load(file)
			model = build_model(HyperparameterTuner.Hyperparameters.from_dict(best['hyperparameters']))
			model.predict(encoded_dataset['train'].take(1).batch(1))
			model.load_weights(join(output_dir, 'best-model.tf'))
			
			# make predictions
			encoded_dataset['test'] = DataUtils.load_from_splits(dataset['test'], data_path, image_size=image_size, model_name=None, quick=False)
			outputs = model.predict(encoded_dataset['test'].batch(inference_batch_size))
			if len(output_config) == 1: outputs = [outputs]
			predictions = [(np.argmax(output, axis=-1) if c == 'x' else np.where(output > 0.5, 1, 0)).tolist() for output, c in zip(outputs, output_config)]
			outputs = [output.tolist() for output in outputs]
			if len(output_config) == 1: outputs = outputs[0]; predictions = predictions[0]
			else: outputs = list(zip(*outputs)); predictions = list(zip(*predictions))
			
			# save predictions
			dataset['test']['output'] = outputs
			dataset['test']['prediction'] = predictions
			dataset['test'].to_json(join(output_dir, 'test-%srun%d-predictions.json' % (str(split) + '-' if split else '', run)), orient='records', indent=1, force_ascii=False)
			
			made_progress = True
		if made_progress: made_progress2 = True
