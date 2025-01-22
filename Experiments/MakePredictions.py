''' Author: Dominik Beese
>>> Use a pre-trained model for making predictions.
<<<
'''

import sys
from os.path import join, exists, normpath, sep
import json

import numpy as np


### CONFIGURATION ###

gpu_devices = [0] # enter gpus to use

RECOMPUTE = True # whether to discard existing predictions

# Clip3-MC
experiment_name = 'Clip3-MC' # the name of the experiment to use
num_labels = 13 # the number of labels the model was trained with
timesteps = 10 # the number of timesteps the model was trained with
output_config = 'x' # output layers, x for softmax, d for sigmoid
model_kwargs = dict() # additional kwargs for to the model
split_name = None # the name of the data splits to train/test on
clip_kwargs = {
	'sample_fps': 10/3, # 10 frames in a 3 second clip
	'sample_length': 3, # 3 second clip
	'sample_count': 1,
	'clip_offset_mult': 5, # 1.5 second overlap
	'clip_offset_add': 0
}

inference_batch_size = 32 # the batch size for inference

# import code and model
sys.path.append(join('..', 'Code'))
from Models import ModelUtils, RNN, CNN_RNN
import GPUUtils, DataUtils

# setup paths
model_path = join(experiment_name)
data_file = join('.', experiment_name, 'Splits', 'run%d.json')
data_path = join('..', 'Data')

# select gpus
GPUUtils.set_visible_devices(gpu_devices)


### PREPARE ###

with open(join(experiment_name, 'Evaluation', 'all.json'), 'r', encoding='UTF-8') as file:
	filenames = json.load(file)
filenames = {folder: [join(*normpath(file).split(sep)[:1]) for file in files] for folder, files in filenames.items()}


### PREDICT ###

for architecture_name, folders in filenames.items():
	architecture = architecture_name.split('-')[1].replace('_', '-')
	for model_name in folders:
		t = model_name.split('-')
		split = t[0] if len(t) > 1 else None
		run = int(t[-1].strip('run'))
		if str(split_name) != str(split): continue
		print()
		print('='*50)
		print('>>', architecture, 'Model', model_name)
		print('='*50)
		
		### Setup ###
		
		# parameters
		token_classification = experiment_name.startswith('Video')
		precompute_features = token_classification
		image_size = ModelUtils.sizeof(architecture)
		cnn_name, rnn_name = ModelUtils.parse(architecture) if precompute_features else (None, None)
		
		# setup model
		output_type = [{'x': 'softmax', 'd': 'sigmoid'}[c] for c in output_config]
		Model = lambda **kwargs: RNN(rnn=rnn_name, **kwargs) if precompute_features else CNN_RNN(architecture=architecture, **kwargs)
		model = Model(output_type=output_type, num_labels=num_labels, timesteps=timesteps, token_classification=token_classification, **model_kwargs)
		output_dir = join(model_path, architecture_name, model_name)
		initialized = False
		
		### Predict ###
		
		# for each case id
		with open(data_file % run, 'r', encoding='UTF-8') as file:
			case_ids = json.load(file)['test']
		for case_id in case_ids:
			output_file = join(output_dir, '%d-predictions.json' % case_id)
			if not RECOMPUTE and exists(output_file): continue
			
			# load data
			dataset, encoded_dataset = DataUtils.load_videos(
				data_path=data_path,
				case_ids=[case_id],
				image_size=image_size,
				model_name=cnn_name,
				quick=True,
				**clip_kwargs
			)
			if dataset.empty: continue
			
			# load weights
			if not initialized:
				model.predict(encoded_dataset.take(1).batch(1))
				model.load_weights(join(output_dir, 'best-model.tf'))
				initialized = True
			
			# make predictions
			outputs = model.predict(encoded_dataset.batch(inference_batch_size))
			if len(output_config) == 1: outputs = [outputs]
			predictions = [(np.argmax(output, axis=-1) if c == 'x' else np.where(output > 0.5, 1, 0)).tolist() for output, c in zip(outputs, output_config)]
			outputs = [output.tolist() for output in outputs]
			if len(output_config) == 1: outputs = outputs[0]; predictions = predictions[0]
			else: outputs = list(zip(*outputs)); predictions = list(zip(*predictions))
			
			# save predictions
			dataset['output'] = outputs
			dataset['prediction'] = predictions
			dataset.to_json(output_file, orient='records', indent=1, force_ascii=False)
