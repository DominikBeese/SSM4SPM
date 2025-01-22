''' Author: Dominik Beese
>>> Implementation of a grid hyperparameter tuner.
<<<
'''

from os import makedirs, urandom, listdir, remove
from os.path import join, exists
from shutil import rmtree

import tensorflow as tf
import numpy as np

import json


### GRID HYPERPARAMETERS ###

class Hyperparameters():
	def __init__(self, start_with_trial=0):
		self.names = list()
		self.values = list()
		self.trial = start_with_trial
	
	def __contains__(self, item):
		return item in self.names
	
	def get(self, name, values=None):
		if name not in self:
			self.names.append(name)
			self.values.append(values)
			return values[0]
		else:
			i = self.names.index(name)
			values = self.values[i]
			length = len(values)
			offset = np.prod([len(v) for v in self.values[:i]]) or 1
			return values[int(self.trial/offset)%length]
	
	def getAll(self):
		return {name: self.get(name) for name in self.names}
	
	def trials(self):
		return np.prod([len(v) for v in self.values])
	
	def nextTrial(self):
		self.trial += 1
	
	@classmethod
	def from_dict(cls, parameters):
		hp = cls()
		for k, v in parameters.items():
			hp.get(k, values=[v])
		return hp


### GRID TUNER ###

class GridTuner():
	def __init__(
		self,
		build_model: callable,
		objective: str = 'val_loss',
		direction: str = 'minimize',
		executions_per_trial: int = 1,
		output_dir: str = None,
		output_predictions: bool = True,
		save_best_model_weights: bool = False,
	):
		""" Configures the grid search.
			build_model - the function that builds and returns a model
			objective - the objective to minimize
			direction - whether to minimize or maximize the objective
			executions_per_trial - the number of executions, i.e. model trainings, to do for each hyperparameter configuration
			output_dir - the directory to save logs, predictions and model weights to
			output_predictions - whether to compute and save predictions
			save_best_model_weights - whether to save or discard the weights of the best performing model
		"""
		if output_dir is not None and exists(output_dir): print('WARNING:', 'Found existing logs, continue search.')
		if output_dir is None: print('WARNING:', 'Not output directory given, nothing will be logged.')
		self.build_model = build_model
		self.objective = objective
		self.extremum = min if direction == 'minimize' else max
		self.executions_per_trial = executions_per_trial
		self.output_dir = output_dir
		self.output_predictions = output_predictions
		self.save_best_model_weights = save_best_model_weights
		try:
			self.load_log()
			self.load_predictions()
			self.initialized = True
		except json.decoder.JSONDecodeError:
			print('WARNING:', 'Existing logs cannot be read, skip search.')
			self.initialized = False
	
	def search(
		self,
		data,
		start_with_trial: int = 1,
		end_with_trial: int = None,
		validate_execution: callable = None,
		retry_execution: callable = None,
		run_after_each_execution: callable = None,
		seed: int = None,
		tensorboard: bool = True,
		**kwargs
	):
		""" Starts the full grid search.
			data - training data
			start_with_trial - skip certain trials from the grid search
			end_with_trial - skip certain trials from the grid search
			validate_execution - function to run before each execution that returns true if the (already finished) execution is valid or not, it gets the history and the current hyperparameters
			retry_execution - function to run after each execution that returns true if the current execution should be discarded and repeated, it gets the model, the history and the current hyperparameters
			run_after_each_execution - function that runs after each execution, it gets the model, the history and the current hyperparameters
			seed - if None, use random seeds, otherwise uses the given seed
				Note: if you use, executions_per_trial > 1, it does not make sense to fix the seed
			tensorboard - whether to log using tensorboard
			**kwargs - for model.fit()
			Returns True if progress was made, i.e., not everything was skipped.
		"""
		if self.output_dir is None: tensorboard = False
		if validate_execution is None: validate_execution = lambda **kwargs: True
		if not self.initialized: return
		
		# batch validation data
		if 'validation_data' in kwargs:
			if 'validation_batch_size' in kwargs:
				kwargs['validation_data'] = kwargs['validation_data'].batch(kwargs['validation_batch_size'])
			elif 'batch_size' in kwargs:
				kwargs['validation_data'] = kwargs['validation_data'].batch(kwargs['batch_size'])
			else: kwargs['validation_data'] = kwargs['validation_data']
		
		# create and init hyperparameters
		start_with_trial = start_with_trial - 1 # trial #1 is 0
		hp = Hyperparameters(start_with_trial)
		_ = self.build_model(hp)
		
		# start grid search
		made_progress = False
		max_trials = hp.trials()
		end_with_trial = max_trials if end_with_trial is None else min(end_with_trial, max_trials)
		best_score = self.extremum((execution['objective'][-1] for trial in self.log or list() for execution in trial.get('executions', list())), default=None)
		for trial in range(start_with_trial, end_with_trial):
			print('Trial %d of %d' % (trial+1, max_trials))
			hyperparameters = hp.getAll()
			print('Hyperparameters:', '{%s}' % ', '.join('%s: %s' % p for p in hyperparameters.items()))
			
			# skip if trial is finished
			if self.output_predictions and len(self.log) != len(self.predictions): raise Exception('Inconsistent logs')
			if trial > len(self.log): raise Exception('Trial from the future: %d' % trial)
			#if trial < len(self.log) and len(self.log[trial]['executions']) >= self.executions_per_trial:
			#	print('Already finished') # Note: not compatible with validate_execution
			#	print()
			#	hp.nextTrial()
			#	continue
			if trial == len(self.log):
				self.log.append({
					'hyperparameters': hyperparameters,
					'best_execution': None,
					'executions': list(),
				})
				if self.output_predictions: self.predictions.append(list())
			
			# update hyperparameters
			if 'batch_size' in hp: kwargs['batch_size'] = hp.get('batch_size')
			if 'epochs' in hp: kwargs['epochs'] = hp.get('epochs')
			
			# multiple executions
			execution = 0
			while execution < self.executions_per_trial:
				# skip if execution is finished
				if self.output_predictions and len(self.log[trial]['executions']) != len(self.predictions[trial]): raise Exception('Inconsistent logs')
				if execution > len(self.log[trial]['executions']): raise Exception('Execution from the future: %d' % execution)
				if execution < len(self.log[trial]['executions']) and 'seed' in self.log[trial]['executions'][execution] and validate_execution(history=self.log[trial]['executions'][execution], hyperparameters=hyperparameters):
					execution += 1
					continue
				if execution == len(self.log[trial]['executions']):
					self.log[trial]['executions'].append(dict())
					if self.output_predictions: self.predictions[trial].append(dict())
				elif self.save_best_model_weights and self.output_dir is not None:
					if self.log[trial]['executions'][execution]['objective'][-1] == best_score:
						print('WARNING:', 'Validate execution deletes the current best model.')
						for filename in listdir(self.output_dir):
							if filename.startswith('best-model.tf'):
								remove(join(self.output_dir, filename))
				
				# set random seed
				current_seed = seed or int.from_bytes(urandom(4), 'little')
				tf.keras.utils.set_random_seed(current_seed)
				
				# build model
				model = self.build_model(hp)
				
				# fit model
				batched_data = data.shuffle(100000, reshuffle_each_iteration=True).batch(kwargs['batch_size']) if 'batch_size' in kwargs else data
				callbacks = list()
				if tensorboard:
					log_dir = join(self.output_dir, 'tensorboard', 'trial%d' % (trial+1), 'exec%d' % (execution+1))
					callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
				callbacks += kwargs.get('callbacks', list())
				history = model.fit(batched_data, callbacks=callbacks, **{k: v for k, v in kwargs.items() if k != 'callbacks'})
				
				# check retry condition
				if retry_execution is not None:
					retries = self.log[trial]['executions'][execution].get('retries', 0)
					if retry_execution(model=model, history=history, hyperparameters=hyperparameters, retries=retries):
						if tensorboard: rmtree(log_dir, ignore_errors=True)
						self.log[trial]['executions'][execution]['retries'] = retries + 1
						self.save_log()
						continue
				
				# evaluate model
				batched_data = data.batch(kwargs['batch_size']) if 'batch_size' in kwargs else data
				for k, v in zip(model.metrics_names, model.evaluate(batched_data)):
					history.history[k].append(v)
				if 'validation_data' in kwargs:
					for k, v in zip(model.metrics_names, model.evaluate(kwargs['validation_data'])):
						history.history['val_'+k] = history.history.get('val_'+k, list()) + [v]
				self.log[trial]['executions'][execution]['seed'] = current_seed
				self.log[trial]['executions'][execution]['objective'] = history.history[self.objective]
				self.log[trial]['executions'][execution]['metrics'] = history.history
				self.save_log()
				if self.output_predictions:
					preds = {'train': model.predict(batched_data).tolist()}
					if 'validation_data' in kwargs: preds['val'] = model.predict(kwargs['validation_data']).tolist()
					self.predictions[trial][execution] = preds
					self.save_predictions()
				
				# save model
				current_score = history.history[self.objective][-1]
				if best_score is None or self.extremum(best_score, current_score) == current_score:
					best_score = current_score
					if self.save_best_model_weights and self.output_dir is not None:
						makedirs(self.output_dir, exist_ok=True)
						model.save_weights(join(self.output_dir, 'best-model.tf'))
				
				# run after each execution
				if run_after_each_execution is not None:
					run_after_each_execution(model=model, history=history, hyperparameters=hyperparameters)
				
				# next execution
				made_progress = True
				execution += 1
			
			# evaluate trial
			best_execution = self.extremum(enumerate(self.log[trial]['executions']), key=lambda x: x[1]['objective'][-1])[0]
			self.log[trial]['best_execution'] = best_execution
			self.save_log()
			self.save_best_configuration()
			print('Best result:', ' - '.join('%s: %.4f' % (m, s[-1]) for m, s in self.log[trial]['executions'][best_execution]['metrics'].items()))
			print()
			
			# next trial
			hp.nextTrial()
		
		# print best hyperparameters
		best_configuration = self.get_best_configuration()
		print('Best hyperparameters:', '{%s}' % ', '.join('%s: %s' % p for p in best_configuration['hyperparameters'].items()))
		print('Best result:', ' - '.join('%s: %.4f' % (m, s[-1]) for m, s in best_configuration['metrics'].items()))
		return made_progress
	
	def get_best_configuration(self):
		if not self.initialized: raise Exception('Not initialized')
		best_trial = self.extremum([t for t in self.log if t['best_execution'] is not None], key=lambda x: x['executions'][x['best_execution']]['objective'][-1])
		best_execution = best_trial['executions'][best_trial['best_execution']]
		return {
			'hyperparameters': best_trial['hyperparameters'],
			'seed': best_execution['seed'],
			'objective': best_execution['objective'],
			'metrics': best_execution['metrics'],
		}
	
	def _save(self, filename, data):
		if self.output_dir is None: return
		makedirs(self.output_dir, exist_ok=True)
		with open(join(self.output_dir, filename), 'w', encoding='UTF-8') as file:
			json.dump(data, file, indent='  ', ensure_ascii=False)
	
	def save_log(self):
		if not self.initialized: raise Exception('Not initialized')
		self._save('log.json', self.log)
	
	def save_best_configuration(self):
		if not self.initialized: raise Exception('Not initialized')
		self._save('best-configuration.json', self.get_best_configuration())
	
	def save_predictions(self):
		if not self.initialized: raise Exception('Not initialized')
		self._save('predictions.json', self.predictions)
	
	def _load(self, filename):
		if self.output_dir is None: return list()
		if not exists(join(self.output_dir, filename)): return list()
		with open(join(self.output_dir, filename), 'r', encoding='UTF-8') as file:
			return json.load(file)
	
	def load_log(self):
		self.log = self._load('log.json')
	
	def load_predictions(self):
		self.predictions = self._load('predictions.json')
