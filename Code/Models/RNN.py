''' Author: Dominik Beese
>>> Implementation of several RNN models inspired by in Cataract-1K (https://www.nature.com/articles/s41597-024-03193-4).
	- LSTM [Hochreiter, 1997]
	- GRU [Cho et al. 2014]
    - Transformer Encoder (https://arxiv.org/pdf/1706.03762)
	- MambaBlock & Mamba (https://arxiv.org/pdf/2312.00752)
		* MambaBlock is a single Mamba block
		* Mamba is a single residual block, i.e. a layer normalization, a Mamba block, and a summation
<<<
'''

from typing import Union

import tensorflow as tf
from . import Transformer, Mamba
import re


class RNN(tf.keras.Model):
	""" A RNN model. """
	
	def __init__(
		self,
		rnn: str,
		rnn_blocks: int = None,
		rnn_layers: int = 2,
		feature_dimension: int = 1000,
		rnn_dimension: int = 1000,
		force_layer_before_rnn_block: bool = False,
		rnn_block_dropout: float = 0.1,
		rnn_block_end_dropout: float = 0.5,
		rnn_layer_dropout: float = 0,
		timesteps: int = None,
		token_classification: bool = False,
		output_type: Union[str, list[str]] = 'softmax',
		num_labels: Union[int, list[int]] = 2,
		rnn_kwargs: dict = None,
		*args, **kwargs
	):
		""" Initializes the CNN-RNN model.
			rnn - the RNN architecture to use
			rnn_blocks - the number of blocks to use from the given architecture, e.g. multiple LSTM layers
			             if None, inferred by the rnn argument
			rnn_layers - the number of fully connected layers after the rnn blocks and before the output layer
			feature_dimension - the dimensionality of the features used as the inputs
			rnn_dimension - the number of units for rnn blocks and fully connected layers
							if rnn_dimension != feature_dimension, add a Dense layer before the first rnn block
			force_layer_before_rnn_block - if True, forces the Dense layer before the first rnn block
			rnn_block_dropout - the dropout rate inbetween individual rnn blocks
			rnn_block_end_dropout - the dropout rate after the rnn blocks
			rnn_layer_dropout - the dropout rate inbetween and after the fully connected layers
			timesteps - if None, uses the last output from the RNN and variable sequence lengths are supported
			            otherwise, uses the full sequence with the given number of timesteps and flattens it
						if token_classification is True, ignores this input
			token_classification - use a token classification output (sequence of output layers) or a sequence classification output (single output layer)
			output_type - the types of output layers: softmax or sigmoid
			num_labels - the number of outputs/labels for each output layer
			rnn_kwargs - additional kwargs for the RNN layer
		"""
		super().__init__(*args, **kwargs)
		if rnn_blocks is None:
			rnn, rnn_blocks = re.match(r'([a-zA-Z]+)([0-9]*)', rnn).groups()
			rnn_blocks = int(rnn_blocks) if rnn_blocks else 1
		if rnn.lower() not in ['lstm', 'bilstm', 'gru', 'bigru', 'transformer', 'mambablock', 'mamba']:
			raise NotImplementedError('RNN architecture not implemented: %s' % rnn)
		if rnn_blocks <= 0: raise ValueError('The number of rnn blocks must be positive')
		if rnn_layers <= 0: raise ValueError('The number of fully connected layers must be positive')
		if feature_dimension <= 0: raise ValueError('The feature dimension must be positive')
		if rnn_dimension <= 0: raise ValueError('The number units must be positive')
		if rnn_block_dropout < 0 or rnn_block_dropout > 1: raise ValueError('The dropout rate must be between zero and one.')
		if rnn_block_end_dropout < 0 or rnn_block_end_dropout > 1: raise ValueError('The dropout rate must be between zero and one.')
		if rnn_layer_dropout < 0 or rnn_layer_dropout > 1: raise ValueError('The dropout rate must be between zero and one.')
		if timesteps is not None and timesteps <= 0: raise ValueError('The number of timesteps must either be None or an positive integer.')
		for t in output_type if isinstance(output_type, list) else [output_type]:
			if t not in ['softmax', 'sigmoid']: raise ValueError('The output type must be softmax or sigmoid.')
		for n in num_labels if isinstance(num_labels, list) else [num_labels]:
			if n <= 0: raise ValueError('The number of output labels must be positive.')
		if len(output_type if isinstance(output_type, list) else [output_type]) != len(num_labels if isinstance(num_labels, list) else [num_labels]): raise ValueError('The same number of output types and number of output labels must be provided.')
		if token_classification is True: timesteps = None
		self.rnn = rnn
		self.rnn_blocks = rnn_blocks
		self.rnn_layers = rnn_layers
		self.feature_dimension = feature_dimension
		self.rnn_dimension = rnn_dimension
		self.force_layer_before_rnn_block = force_layer_before_rnn_block
		self.rnn_block_dropout = rnn_block_dropout
		self.rnn_block_end_dropout = rnn_block_end_dropout
		self.rnn_layer_dropout = rnn_layer_dropout
		self.timesteps = timesteps
		self.token_classification = token_classification
		self.output_type = output_type if isinstance(output_type, list) else [output_type]
		self.num_labels = num_labels if isinstance(num_labels, list) else [num_labels]
		self.rnn_kwargs = rnn_kwargs or dict()
	
	def build(self, input_shape):
		""" Builds the model.
			The input shape is in the format (None, timesteps, features).
		"""
		# RNN Helper
		def BiLSTM(units, **kwargs):
			return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, **kwargs))
		def BiGRU(units, **kwargs):
			return tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, **kwargs))
		def TransformerEncoder(units, **kwargs):
			kwargs_rest = {k: v for k, v in kwargs.items() if k != 'inner_dim'}
			return Transformer.TransformerEncoderBlock(inner_dim=kwargs.get('inner_dim', units), **kwargs_rest)
		def MambaBlock(units, **kwargs):
			kwargs_rest = {k: v for k, v in kwargs.items() if k != 'd_model'}
			return Mamba.MambaBlock(d_model=kwargs.get('d_model', units), **kwargs_rest)
		def ResidualBlock(units, **kwargs):
			if 'd_model' in kwargs: raise ValueError('d_model cannot be set when using residual blocks')
			##if units != self.feature_dimension: raise ValueError('Must use as many units as the feature dimension when using residual blocks') # not anymore, because of feature dimension
			return Mamba.ResidualBlock(d_model=units, **kwargs)
		
		# RNN
		if self.rnn.lower() == 'lstm': rnn_layer = tf.keras.layers.LSTM
		elif self.rnn.lower() == 'bilstm': rnn_layer = BiLSTM
		elif self.rnn.lower() == 'gru': rnn_layer = tf.keras.layers.GRU
		elif self.rnn.lower() == 'bigru': rnn_layer = BiGRU
		elif self.rnn.lower() == 'transformer': rnn_layer = TransformerEncoder
		elif self.rnn.lower() == 'mambablock': rnn_layer = MambaBlock
		elif self.rnn.lower() == 'mamba': rnn_layer = ResidualBlock
		else: raise Exception()
		x = inputs = tf.keras.layers.Input(shape=(self.timesteps, self.feature_dimension)) # IN: (timesteps, features)
		if self.force_layer_before_rnn_block is True or self.rnn_dimension != self.feature_dimension: x = tf.keras.layers.Dense(self.rnn_dimension, activation='relu')(x) # modify feature dimension
		for _ in range(self.rnn_blocks-1):
			x = rnn_layer(self.rnn_dimension, return_sequences=True, **self.rnn_kwargs)(x)
			if self.rnn_block_dropout > 0: x = tf.keras.layers.Dropout(self.rnn_block_dropout)(x)
		return_sequences = self.token_classification or self.timesteps is not None # return last output or all outputs
		x = rnn_layer(self.rnn_dimension, return_sequences=return_sequences, **self.rnn_kwargs)(x)
		if self.rnn_block_end_dropout > 0: x = tf.keras.layers.Dropout(self.rnn_block_end_dropout)(x)
		if not self.token_classification and return_sequences: x = tf.keras.layers.Flatten()(x) # flatten matrix to sequence
		for _ in range(self.rnn_layers):
			x = tf.keras.layers.Dense(self.rnn_dimension, activation='relu')(x)
			if self.rnn_layer_dropout > 0: x = tf.keras.layers.Dropout(self.rnn_layer_dropout)(x)
		outputs = list()
		for t, n in zip(self.output_type, self.num_labels):
			if t == 'softmax':
				outputs.append(tf.keras.layers.Dense(n, activation='softmax')(x)) # OUT: (softmax,)
			elif t == 'sigmoid':
				outputs.append(tf.keras.layers.Dense(n, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(x)) # OUT: (sigmoid,)
			else: raise Exception()
		self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.rnn)
	
	def call(self, inputs, **kwargs):
		""" Calls the model. """
		return self.model(inputs)
