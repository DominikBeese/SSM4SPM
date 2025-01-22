''' Author: Dominik Beese
>>> Implementation of tensorflow losses.
<<<
'''

import tensorflow as tf


### WEIGHTED LOSSES ###

class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
	""" Implementation of a weighted tf.keras.losses.SparseCategoricalCrossentropy. """
	
	def __init__(self, class_weight, dtype=None, name='weighted_sparse_categorical_crossentropy', **kwargs):
		super().__init__(name=name, **kwargs)
		self.dtype = dtype or tf.keras.backend.floatx()
		self.class_weight = tf.cast(class_weight, self.dtype)
	
	def call(self, y_true, y_pred):
		scc = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
		weights = tf.gather(self.class_weight, y_true)
		return scc * weights
	
	def get_config(self):
		config = {'weighted': True}
		base_config = loss.get_config()
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		raise NotImplementedError()

class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
	""" Implementation of a weighted tf.keras.losses.BinaryCrossentropy. """
	
	def __init__(self, class_weight=None, dtype=None, name='weighted_binary_crossentropy', **kwargs):
		super().__init__(name=name, **kwargs)
		self.dtype = dtype or tf.keras.backend.floatx()
		self.class_weight = tf.cast(class_weight, self.dtype) if class_weight is not None else None
		
	def call(self, y_true, y_pred):
		y_true = tf.cast(y_true, y_pred.dtype)
		bc = tf.keras.backend.binary_crossentropy(y_true, y_pred)
		if self.class_weight is not None:
			weights = y_true * self.class_weight + 1
			bc = bc * weights
		return tf.keras.backend.mean(bc)
	
	def get_config(self):
		config = {'weighted': True}
		base_config = loss.get_config()
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		raise NotImplementedError()


### WEIGHTED WRAPPER ###

class Weighted(tf.keras.losses.Loss):
	""" Implementation of a generic wrapper for a weighted loss function. """
	
	def __init__(self, loss, class_weight, dtype=None, **kwargs):
		super().__init__(**kwargs)
		self.dtype = dtype or tf.keras.backend.floatx()
		self.loss = loss
		self.class_weight = tf.cast(class_weight, self.dtype)
	
	def call(self, y_true, y_pred):
		return self.loss.call(y_true, y_pred) * tf.gather(self.class_weight, y_true)
	
	def get_config(self):
		config = {'weighted': True}
		base_config = loss.get_config()
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		raise NotImplementedError()
