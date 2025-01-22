''' Author: Dominik Beese
>>> Trainer for the experiment.
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

# uncomment the desired experiment
""" # Clip-1vR
architectures = [c+'-'+r for c in ['ResNet50', 'VGG16'] for r in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Clip-1vR' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = ['CP', 'Cap', 'Hyd', 'IA', 'Inc', 'LI', 'LP', 'Pha', 'TA', 'VS', 'Vis-ACF'] # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [16],
	'learning_rate': [5e-5, 3e-5, 2e-5],
}
validation_batch_size = 16 #"""

""" # Clip-MC (normal)
architectures = [c+'-'+r for c in ['VGG16', 'ResNet50', 'InceptionV3', 'DenseNet121', 'EfficientNetB0'] for r in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Clip-MC' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [16],
	'learning_rate': [5e-5, 3e-5, 2e-5],
}
validation_batch_size = 16 #"""

""" # Clip-MC (large)
architectures = [c+'-'+r for c in ['VGG19', 'ResNet152', 'DenseNet201', 'EfficientNetB2'] for r in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Clip-MC' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [2e-5, 1e-5, 7e-6],
}
validation_batch_size = 8 #"""

""" # Clip3-MC (large)
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Clip3-MC' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [2e-5, 1e-5, 7e-6],
}
validation_batch_size = 8 #"""

""" # Video4-MC (1-Layer)
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Video4-MC' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = {'rnn_layer_dropout': 0.5} # additional kwargs for to the model
split_names = [3, 15, 60, 180] # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [16],
	'learning_rate': [7e-6, 5e-6, 3e-6, 2e-6],
}
validation_batch_size = 16 #"""

""" # Video4-MC (3-Layer)
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['LSTM3', 'GRU3', 'BiLSTM3', 'BiGRU3', 'Transformer3', 'Mamba3']]
experiment_name = 'Video4-MC' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = {'rnn_layer_dropout': 0.5} # additional kwargs for to the model
split_names = [3, 15, 60, 180] # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [16],
	'learning_rate': [3e-5, 2e-5, 1e-5],
}
validation_batch_size = 16 #"""

""" # Clip2-IA (large)
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Clip2-IA' # the name of the experiment to train for
output_config = 'd' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [2e-6, 1e-6, 7e-7],
}
validation_batch_size = 8 #"""

""" # Clip5-P (large) [needs extra]
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['Transformer']]
experiment_name = 'Clip5-PIA' # the name of the experiment to train for
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [5e-6, 3e-6, 2e-6],
}
validation_batch_size = 8 #"""

""" # Clip5-IA (large) [needs extra]
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['Transformer']]
experiment_name = 'Clip5-PIA' # the name of the experiment to train for
output_config = 'd' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [2e-6, 1e-6, 7e-7],
}
validation_batch_size = 8 #"""

""" # Clip5-PIA (large) [needs extra]
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['Transformer']]
experiment_name = 'Clip5-PIA' # the name of the experiment to train for
output_config = 'xd' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [3e-6, 2e-6, 1e-6],
}
validation_batch_size = 8 #"""

""" # Clip5-IA+P (large) [needs double extra]
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['Transformer']]
experiment_name = 'Clip5-PIA' # the name of the experiment to train for
finetuned_features = 'ResNet152_Transformer-Clip5_IA-run%d'
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1, 2, 3, 4, 5] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [7e-5, 5e-5, 3e-5, 2e-5, 1e-5, 7e-6, 5e-6, 3e-6, 2e-6, 1e-6, 7e-7, 5e-7, 3e-7, 2e-7, 1e-7],
}
validation_batch_size = 8 #"""

""" # Full-I [needs quick=False and Extra]
architectures = [c+'-'+r for c in ['ResNet152'] for r in ['BiLSTM', 'Transformer', 'Mamba', 'GRU', 'LSTM', 'BiGRU']] #['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Transformer', 'Mamba']]
experiment_name = 'Full-I' # the name of the experiment to train for
finetuned_features = 'ResNet152_BiLSTM-Clip2_IA-run%d'
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = {'feature_dimension': 1000, 'rnn_dimension': 512, 'rnn_layer_dropout': 0.5} # additional kwargs for to the model
split_names = None # the name of the data splits to train/test on
split_runs = [1] # the runs to do, e.g. [1, 2, 3]
hyperparameters = {
	'batch_size': [8],
	'learning_rate': [2e-7, 1e-7, 7e-8],
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
sys.path.append(join('..', 'Code'))
from Models import ModelUtils, RNN, CNN_RNN
import GPUUtils, DataUtils
from Training import HyperparameterTuner, Losses, Metrics, Callbacks

# setup paths
data_file = join('.', experiment_name, 'Splits', '%s-run%d.json')
data_path = join('..', 'Data')

# select gpus
GPUUtils.set_visible_devices(gpu_devices)

"""
### Clip5-PIA EXTRA ###

def preprocess(data):
	if output_config == 'xd': return data
	for df in data.values():
		if output_config == 'x': df['label'] = df['label'].map(lambda x: x[0])
		elif output_config == 'd': df['label'] = df['label'].map(lambda x: x[1])
		else: raise ValueError()
	return data
"""


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
			token_classification = experiment_name.startswith('Video')
			precompute_features = token_classification
			image_size = ModelUtils.sizeof(architecture)
			cnn_name, rnn_name = ModelUtils.parse(architecture) if precompute_features else (None, None)
			
			# load data
			dataset = {k: pd.read_json(data_file % (k + ('-' + str(split) if split else ''), run)) for k in ['train', 'dev', 'test']}
			##dataset = preprocess(dataset) # Clip5-PIA Extra
			encoded_dataset = {k: DataUtils.load_from_splits(dataset[k], data_path, image_size=image_size, model_name=cnn_name, quick=True) for k in ['train', 'dev']}
			
			### Model Builder ###
			
			def build_model(hp):
				# get parameters
				hp.get('batch_size', values=hyperparameters['batch_size'])
				hp.get('epochs', values=hyperparameters['epochs'])
				learning_rate = hp.get('learning_rate', values=hyperparameters['learning_rate'])
				
				# build model
				output_type = [{'x': 'softmax', 'd': 'sigmoid'}[c] for c in output_config]
				if len(output_config) > 1:
					labels = dataset['train']['label'].apply(pd.Series)
					labels = [labels[i] for i in labels.columns]
				else: labels = [dataset['train']['label']]
				labels = [l.explode().dropna().to_numpy() if c == 'x' else np.where(pd.DataFrame(l.to_list()).to_numpy() == 1)[1] for l, c in zip(labels, output_config)]
				num_labels = [np.max(l)+1 for l in labels]
				class_weight = [sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(l), y=l) for l in labels]
				timesteps = dataset['train']['frames'].map(len).unique().max()
				Model = lambda **kwargs: RNN(rnn=rnn_name, **kwargs) if precompute_features else CNN_RNN(architecture=architecture, **kwargs)
				model = Model(output_type=output_type, num_labels=num_labels, timesteps=timesteps, token_classification=token_classification, **model_kwargs)
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
			encoded_dataset['test'] = DataUtils.load_from_splits(dataset['test'], data_path, image_size=image_size, model_name=cnn_name, quick=False)
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
