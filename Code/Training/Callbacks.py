''' Author: Dominik Beese
>>> Implementation of tensorflow callbacks.
<<<
'''

import tensorflow as tf


class EarlyStopping(tf.keras.callbacks.Callback):
	""" Callback that stops training if tbe monitored metric has stopped improving. """
	
	def __init__(
		self,
		target: str = 'val_loss',
		min_delta: int = 0,
		patience: int = 0,
		mode: str = 'min',
		earliest_stop_epoch: int = 0,
		restore_best_weights: bool = True,
	):
		""" If the [target] has not improved by at least [min_delta] after [patience] epochs, stop training.
			Stops the model no earlier than [earliest_stop_epoch], but possibly in this very epoch.
			If [restore_best_weights] is True, the weights of the best performing model in any(!) epoch are restored.
		"""
		super().__init__()
		self.target = target
		self.min_delta = abs(min_delta)
		self.patience = patience
		self.mode = mode
		self.earliest_stop_epoch = earliest_stop_epoch
		self.restore_best_weights = restore_best_weights
		
		self.best = float('inf') if self.mode == 'min' else -float('inf')
		self.best_weights = None
		self.wait = 0
		self.stopped_epoch = None
	
	def on_train_begin(self, logs=None):
		self.best = float('inf') if self.mode == 'min' else -float('inf')
		self.best_weights = None
		self.wait = 0
		self.stopped_epoch = None
	
	def on_epoch_end(self, epoch, logs=None):
		if logs is None or self.target not in logs:
			self.wait += 1
			return
		if self.restore_best_weights and self.best_weights is None:
			self.best_weights = self.model.get_weights()
		current = logs[self.target]
		if (self.mode == 'max' and current - self.min_delta > self.best) or (self.mode == 'min' and current + self.min_delta < self.best):
			self.best = current
			self.wait = 0
			if self.restore_best_weights:
				self.best_weights = self.model.get_weights()
			return
		self.wait += 1
		if self.wait < self.patience or epoch == 0: return
		if epoch + 1 < self.earliest_stop_epoch: return
		print()
		print(self.__class__.__name__ + ':', 'stop training')
		self.stopped_epoch = epoch
		self.model.stop_training = True
		self.model.stopped_by = self.__class__.__name__
	
	def on_train_end(self, logs=None):
		if self.restore_best_weights and self.best_weights is not None:
			print(self.__class__.__name__ + ':', 'restore best weights')
			self.model.set_weights(self.best_weights)


class EarlyFailing(tf.keras.callbacks.Callback):
	""" Callback that stops training if at the end of the given epoch, the given measurement has not reached a given limit. """
	
	def __init__(
		self,
		epoch: int,
		target: str,
		value: float,
		mode: str = 'max',
	):
		""" If at epoch [epoch], the [target] is not [direction] that [value], stop training. """
		super().__init__()
		self.epoch = epoch
		self.target = target
		self.value = value
		self.mode = mode
		
		self.stopped_epoch = None
		self.delay = 0
	
	def on_train_begin(self, logs=None):
		self.stopped_epoch = None
		self.delay = 0
	
	def on_epoch_end(self, epoch, logs=None):
		if epoch + 1 != self.epoch + self.delay: return
		if logs is None or self.target not in logs:
			self.delay += 1
			return
		if self.mode == 'max':
			if logs[self.target] > self.value: return
		elif self.mode == 'min':
			if logs[self.target] < self.value: return
		else: raise Exception('Unknown mode: %s' % self.mode)
		print()
		print(self.__class__.__name__ + ':', 'stop training')
		self.stopped_epoch = epoch
		self.model.stop_training = True
		self.model.stopped_by = self.__class__.__name__
