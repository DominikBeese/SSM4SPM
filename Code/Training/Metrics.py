''' Author: Dominik Beese
>>> Implementation of tensorflow metrics.
<<<
'''

import tensorflow as tf


### BASE METRICS ###

class F1Score(tf.keras.metrics.Metric):
	""" Implementation of a (Macro) F1 Score as a keras metric.
		Note: Very slow and inefficient for sequences.
	"""
	
	def __init__(self, num_labels=2, average='macro', name='f1score', dtype=None):
		super().__init__(name=name, dtype=dtype)
		if average != 'macro': raise NotImplementedError()
		self.num_labels = num_labels
		self.average = average
		self._build()
	
	def _build(self):
		def _add_zeros_variable(name):
			return self.add_weight(
				name=name,
				shape=(self.num_labels,),
				initializer=tf.keras.initializers.Zeros(),
				dtype=self.dtype,
			)
		self.true_positives = _add_zeros_variable('true_positives')
		self.false_positives = _add_zeros_variable('false_positives')
		self.false_negatives = _add_zeros_variable('false_negatives')
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		matrix = tf.reshape(tf.repeat(tf.range(self.num_labels, dtype=self.dtype), len(y_true)), (self.num_labels, -1))
		def measure(yt, yp):
			self.true_positives.assign_add(
				tf.reduce_sum(
					tf.where(yt == matrix, 1., 0.) * tf.where(yp == matrix, 1., 0.)
				, axis=1)
			)
			self.false_positives.assign_add(
				tf.reduce_sum(
					(1 - tf.where(yt == matrix, 1., 0.)) * tf.where(yp == matrix, 1., 0.)
				, axis=1)
			)
			self.false_negatives.assign_add(
				tf.reduce_sum(
					tf.where(yt == matrix, 1., 0.) * (1 - tf.where(yp == matrix, 1., 0.))
				, axis=1)
			)
		if len(y_true.shape) > 1:
			for yt, yp in zip(tf.unstack(y_true, axis=1), tf.unstack(y_pred, axis=1)): measure(yt, yp)
		else: measure(y_true, y_pred)
	
	def result(self):
		numerator = 2 * self.true_positives
		denominator = numerator + self.false_positives + self.false_negatives
		f1_score = tf.cast(numerator, dtype=self.dtype) / (tf.cast(denominator, dtype=self.dtype) + tf.keras.backend.epsilon())
		return tf.reduce_mean(f1_score)
	
	def get_config(self):
		config = {
			'name': self.name,
			'dtype': self.dtype,
			'average': self.average,
			'num_labels': self.num_labels,
		}
		base_config = super().get_config()
		return {**base_config, **config}
	
	def reset_state(self):
		for w in self.weights:
			w.assign(tf.zeros(w.shape, dtype=w.dtype))


### DERIVED METRICS ###

class CategoricalF1Score(F1Score):
	""" Implementation of a Categorical (Macro) F1 Score as a keras metric. """
	
	def __init__(self, num_labels=2, average='macro', name='categorical_f1score', dtype=None):
		super().__init__(num_labels=num_labels, average=average, name=name, dtype=dtype)
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		super().update_state(
			tf.cast(tf.argmax(y_true, axis=-1), dtype=y_true.dtype),
			tf.cast(tf.argmax(y_pred, axis=-1), dtype=y_pred.dtype),
			sample_weight
		)

class SparseCategoricalF1Score(F1Score):
	""" Implementation of a Sparse Categorical (Macro) F1 Score as a keras metric. """
	
	def __init__(self, num_labels=2, average='macro', name='categorical_f1score', dtype=None):
		super().__init__(num_labels=num_labels, average=average, name=name, dtype=dtype)
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		dtype = y_pred.dtype
		y_pred = tf.argmax(y_pred, axis=-1)
		if y_true.shape != y_pred.shape: y_true = tf.squeeze(y_true, -1)
		super().update_state(
			tf.cast(y_true, dtype=dtype),
			tf.cast(y_pred, dtype=dtype),
			sample_weight
		)

class CategoricalMacroF1Score(CategoricalF1Score):
	def __init__(self, num_labels=2, name='categorical_f1score', dtype=None):
		super().__init__(num_labels=num_labels, average='macro', name=name, dtype=dtype)

class SparseCategoricalMacroF1Score(SparseCategoricalF1Score):
	def __init__(self, num_labels=2, name='categorical_f1score', dtype=None):
		super().__init__(num_labels=num_labels, average='macro', name=name, dtype=dtype)
