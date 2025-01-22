''' Author: Dominik Beese
>>> Implementation of a CNN architecture with a simple classifier head.
<<<
'''

from typing import Union

import tensorflow as tf
from . import CNN


class CNNForFineTuning(tf.keras.Model):
	""" A CNN architecture with a simple classifier head.
		If possible, the CNN part is pre-trained on ImageNet.
	"""
	
	def __init__(
		self,
		architecture: str = None,
		cnn: str = None,
		feature_dimension: int = 1000,
		head_dimension: int = 1000,
		output_type: Union[str, list[str]] = 'softmax',
		num_labels: Union[int, list[int]] = 2,
		cnn_kwargs: dict = None,
		*args, **kwargs
	):
		""" Initializes the CNN-RNN model.
			architecture - the CNN architecture to use
			               if given, overwrites the cnn argument
			cnn - the CNN architecture to use
			feature_dimension - the dimensionality of the features used as the inputs
			head_dimension - the number of units for rnn blocks and fully connected layers
							if head_dimension != feature_dimension, add a Dense layer before the first rnn block
			output_type - the types of output layers: softmax or sigmoid
			num_labels - the number of outputs/labels for each output layer
			cnn_kwargs - additional kwargs for the CNN application
		"""
		super().__init__(*args, **kwargs)
		if architecture is None: architecture = cnn
		if feature_dimension <= 0: raise ValueError('The feature dimension must be positive')
		if head_dimension <= 0: raise ValueError('The number units must be positive')
		for t in output_type if isinstance(output_type, list) else [output_type]:
			if t not in ['softmax', 'sigmoid']: raise ValueError('The output type must be softmax or sigmoid.')
		for n in num_labels if isinstance(num_labels, list) else [num_labels]:
			if n <= 0: raise ValueError('The number of output labels must be positive.')
		if len(output_type if isinstance(output_type, list) else [output_type]) != len(num_labels if isinstance(num_labels, list) else [num_labels]): raise ValueError('The same number of output types and number of output labels must be provided.')
		self.architecture = architecture
		self.cnn_kwargs = cnn_kwargs
		self.feature_dimension = feature_dimension
		self.head_dimension = head_dimension
		self.output_type = output_type if isinstance(output_type, list) else [output_type]
		self.num_labels = num_labels if isinstance(num_labels, list) else [num_labels]
	
	def build(self, input_shape):
		""" Builds the model.
			The input shape is in the format (None, timesteps, features).
		"""
		# CNN
		cnn = CNN(
			cnn=self.architecture,
			cnn_kwargs=self.cnn_kwargs,
		)
		cnn.build(input_shape)
		
		# Head
		x = cnn.layers[-1].output
		if self.head_dimension != self.feature_dimension: x = tf.keras.layers.Dense(self.head_dimension, activation='relu')(x) # modify feature dimension
		self.cnn = tf.keras.Model(inputs=cnn.layers[0].inputs, outputs=x, name=self.architecture)
		outputs = list()
		for t, n in zip(self.output_type, self.num_labels):
			if t == 'softmax':
				outputs.append(tf.keras.layers.Dense(n, activation='softmax')(x)) # OUT: (softmax,)
			elif t == 'sigmoid':
				outputs.append(tf.keras.layers.Dense(n, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(x)) # OUT: (sigmoid,)
			else: raise Exception()
		self.model = tf.keras.Model(inputs=cnn.layers[0].inputs, outputs=outputs, name=self.architecture)
	
	def call(self, inputs, **kwargs):
		""" Calls the model. """
		return self.model(inputs)
