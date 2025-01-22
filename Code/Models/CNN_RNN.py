''' Author: Dominik Beese
>>> Implementation of a CNN-RNN architecture.
<<<
'''

from typing import Union

import tensorflow as tf
from . import CNN, RNN, ModelUtils


class CNN_RNN(tf.keras.Model):
	""" A CNN-RNN architecture.
		If possible, the CNN part is pre-trained on ImageNet.
	"""
	
	def __init__(
		self,
		architecture: str = None,
		cnn: str = None,
		rnn: str = None,
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
		cnn_kwargs: dict = None,
		rnn_kwargs: dict = None,
		*args, **kwargs
	):
		""" Initializes the CNN-RNN model.
			architecture - the CNN-RNN architecture to use
			               if given, overwrites the cnn and rnn arguments
			cnn - the CNN architecture to use
			rnn - the RNN architecture to use
			rnn_blocks - the number of blocks to use from the given architecture, e.g. multiple LSTM layers
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
			cnn_kwargs - additional kwargs for the CNN application
			rnn_kwargs - additional kwargs for the RNN layer
		"""
		super().__init__(*args, **kwargs)
		if architecture is not None: cnn, rnn = ModelUtils.parse(architecture)
		self.cnn = CNN(
			cnn=cnn,
			cnn_kwargs=cnn_kwargs,
		)
		self.rnn = RNN(
			rnn=rnn,
			rnn_blocks=rnn_blocks,
			rnn_layers=rnn_layers,
			feature_dimension=feature_dimension,
			rnn_dimension=rnn_dimension,
			force_layer_before_rnn_block=force_layer_before_rnn_block,
			rnn_block_dropout=rnn_block_dropout,
			rnn_block_end_dropout=rnn_block_end_dropout,
			rnn_layer_dropout=rnn_layer_dropout,
			timesteps=timesteps,
			token_classification=token_classification,
			output_type=output_type,
			num_labels=num_labels,
			rnn_kwargs=rnn_kwargs,
		)
	
	def call(self, inputs, **kwargs):
		""" Calls the model.
			The CNN is called for each timestep-element.
			The RNN is called with all timestep-elements at once.
		"""
		# CNN
		features = tf.transpose(tf.convert_to_tensor([self.cnn(x) for x in tf.unstack(inputs, axis=1)]), [1, 0, 2])
		
		# RNN
		outputs = self.rnn(features)
		return outputs
