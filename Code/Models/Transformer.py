''' Author: Dominik Beese
>>> Implementation of a Transformer Encoder by The TensorFlow Authors, edited by me.
	* Transformer (https://arxiv.org/pdf/1706.03762)
	* Implementation (https://github.com/tensorflow/models/blob/v2.15.0/official/nlp/modeling/layers/transformer_encoder_block.py#L24-L412)
<<<
'''

import tensorflow as tf
import inspect


### HELPER ###

def filter_kwargs(kwargs):
	denylist = [
		'num_attention_heads', 'intermediate_size', 'intermediate_activation',
		'inner_dim', 'inner_activation', 'output_range', 'kernel_initializer',
		'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
		'activity_regularizer', 'kernel_constraint', 'bias_constraint',
		'use_bias', 'norm_first', 'norm_epsilon', 'output_dropout',
		'attention_dropout', 'inner_dropout', 'attention_initializer',
		'attention_axes', 'share_rezero'
	]
	for unused_key in denylist:
		kwargs.pop(unused_key, None)

def clone_initializer(initializer):
	if isinstance(initializer, tf.initializers.Initializer):
		return initializer.__class__.from_config(initializer.get_config())
	return initializer

def serialize_initializer(initializer, use_legacy_format=False):
	if 'use_legacy_format' in inspect.getfullargspec(tf.initializers.serialize).args:
		return tf.initializers.serialize(initializer, use_legacy_format=use_legacy_format)
	else:
		return tf.initializers.serialize(initializer)

def serialize_regularizer(regularizer, use_legacy_format=False):
	if 'use_legacy_format' in inspect.getfullargspec(tf.regularizers.serialize).args:
		return tf.regularizers.serialize(regularizer, use_legacy_format=use_legacy_format)
	else:
		return tf.regularizers.serialize(regularizer)

def serialize_constraint(constraint, use_legacy_format=False):
	if 'use_legacy_format' in inspect.getfullargspec(tf.constraints.serialize).args:
		return tf.constraints.serialize(constraint, use_legacy_format=use_legacy_format)
	else:
		return tf.constraints.serialize(constraint)


### TRANSFORMER ENCODER BLOCK ###

class TransformerEncoderBlock(tf.keras.layers.Layer):
	""" Implementation of a Transformer Encoder Block.
		This layer implements the Transformer Encoder from "Attention Is All You Need" (https://arxiv.org/abs/1706.03762), which combines a `tf.keras.layers.MultiHeadAttention` layer with a two-layer feedforward network.
	"""
	
	def __init__(self,
		inner_dim,
		num_attention_heads=8,
		inner_activation='relu',
		return_sequences=True,
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros',
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		use_bias=True,
		norm_first=False,
		norm_epsilon=1e-12,
		output_dropout=0.0,
		attention_dropout=0.0,
		inner_dropout=0.0,
		attention_initializer=None,
		attention_axes=None,
		use_query_residual=True,
		key_dim=None,
		value_dim=None,
		output_last_dim=None,
		diff_q_kv_att_layer_norm=False,
		return_attention_scores=False,
		**kwargs
	):
		""" Initializes the Transformer Encoder Block.
			inner_dim - the output dimension of the first Dense layer in a two-layer feedforward network, i.e. d_ff in the paper, e.g. 2048, 4096
			num_attention_heads - number of attention heads, i.e. h in the paper, e.g. 8, 16
			inner_activation - the activation for the first Dense layer in a two-layer feedforward network, i.e. `relu` in the paper
			return_sequences - whether to return the last output in the output sequence, or the full sequence
			kernel_initializer - initializer for dense layer kernels
			bias_initializer - initializer for dense layer biases
			kernel_regularizer - regularizer for dense layer kernels
			bias_regularizer - regularizer for dense layer biases
			activity_regularizer - regularizer for dense layer activity
			kernel_constraint - constraint for dense layer kernels
			bias_constraint - constraint for dense layer kernels
			use_bias - whether to enable use_bias in attention layer
			   if set False, use_bias in attention layer is disabled
			norm_first - whether to normalize inputs to attention and intermediate dense layers
			   if set False, output of attention and intermediate dense layers is normalized
			norm_epsilon - epsilon value to initialize normalization layers
			output_dropout - dropout probability for the post-attention and output dropout
			attention_dropout - dropout probability for within the attention layer
			inner_dropout - dropout probability for the first Dense layer in a two-layer feedforward network
			attention_initializer - initializer for kernels of attention layers
			   if set `None`, attention layers use kernel_initializer as initializer for kernel
			attention_axes - axes over which the attention is applied
			   `None` means attention over all axes, but batch, heads, and features
			use_query_residual - toggle to execute residual connection after attention
			key_dim - `key_dim` for the `tf.keras.layers.MultiHeadAttention`
			   If `None`, we use the first `input_shape`'s last dim
			value_dim - `value_dim` for the `tf.keras.layers.MultiHeadAttention`
			output_last_dim - final dimension of the output of this module
			   this also dictates the value for the final dimension of the multi-head-attention
			   when it's `None`, we use, in order of decreasing precedence, `key_dim` * `num_heads` or the first `input_shape`'s last dim as the output's last dim
			diff_q_kv_att_layer_norm - if `True`, create a separate attention layer norm layer for query and key-value if `norm_first` is `True`
			   invalid to set to `True` if `norm_first` is `False`
			return_attention_scores - if `True`, the output of this layer will be a tuple and additionally contain the attention scores in the shape of `[batch_size, num_attention_heads, seq_dim, seq_dim]`
			**kwargs - keyword arguments
		"""
		filter_kwargs(kwargs)
		super().__init__(**kwargs)
		
		self._num_heads = num_attention_heads
		self._inner_dim = inner_dim
		self._inner_activation = inner_activation
		self._return_sequences = return_sequences
		self._attention_dropout_rate = attention_dropout
		self._output_dropout_rate = output_dropout
		self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self._bias_initializer = tf.keras.initializers.get(bias_initializer)
		self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
		self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self._bias_constraint = tf.keras.constraints.get(bias_constraint)
		self._use_bias = use_bias
		self._norm_first = norm_first
		self._norm_epsilon = norm_epsilon
		self._inner_dropout = inner_dropout
		self._use_query_residual = use_query_residual
		self._key_dim = key_dim
		self._value_dim = value_dim
		self._output_last_dim = output_last_dim
		self._diff_q_kv_att_layer_norm = diff_q_kv_att_layer_norm
		self._return_attention_scores = return_attention_scores
		if attention_initializer: self._attention_initializer = tf.keras.initializers.get(attention_initializer)
		else: self._attention_initializer = clone_initializer(self._kernel_initializer)
		self._attention_axes = attention_axes
		
		if self._diff_q_kv_att_layer_norm and not self._norm_first:
			raise ValueError('Setting `diff_q_and_kv_attention_layer_norm` to True when `norm_first` is False is invalid.')
	
	def build(self, input_shape):
		if isinstance(input_shape, tf.TensorShape): input_tensor_shape = input_shape
		elif isinstance(input_shape, (list, tuple)): input_tensor_shape = tf.TensorShape(input_shape[0])
		else: raise ValueError('The type of input shape argument is not supported, got: %s' % type(input_shape))
		einsum_equation = 'abc,cd->abd'
		if len(input_tensor_shape.as_list()) > 3: einsum_equation = '...bc,cd->...bd'
		hidden_size = input_tensor_shape[-1]
		if hidden_size % self._num_heads != 0: raise ValueError('The input size (%d) is not a multiple of the number of attention heads (%d)' % (hidden_size, self._num_heads))
		if self._key_dim is None: self._key_dim = int(hidden_size // self._num_heads)
		if self._output_last_dim is None: last_output_shape = hidden_size
		else: last_output_shape = self._output_last_dim
		
		common_kwargs = dict(
			bias_regularizer=self._bias_regularizer,
			activity_regularizer=self._activity_regularizer,
			kernel_constraint=self._kernel_constraint,
			bias_constraint=self._bias_constraint
		)
		self._attention_layer = tf.keras.layers.MultiHeadAttention(
			num_heads=self._num_heads,
			key_dim=self._key_dim,
			value_dim=self._value_dim,
			dropout=self._attention_dropout_rate,
			use_bias=self._use_bias,
			kernel_initializer=self._attention_initializer,
			bias_initializer=clone_initializer(self._bias_initializer),
			attention_axes=self._attention_axes,
			output_shape=self._output_last_dim,
			name='self_attention',
			**common_kwargs
		)
		self._attention_dropout = tf.keras.layers.Dropout(
			rate=self._attention_dropout_rate
		)
		self._attention_layer_norm = tf.keras.layers.LayerNormalization(
			name='self_attention_layer_norm',
			axis=-1,
			epsilon=self._norm_epsilon,
			dtype=tf.float32
		)
		self._attention_layer_norm_kv = self._attention_layer_norm
		if self._diff_q_kv_att_layer_norm:
			self._attention_layer_norm_kv = tf.keras.layers.LayerNormalization(
				name='self_attention_layer_norm_kv',
				axis=-1,
				epsilon=self._norm_epsilon,
				dtype=tf.float32
			)
		
		self._intermediate_dense = tf.keras.layers.EinsumDense(
				einsum_equation,
				output_shape=(None, self._inner_dim),
				bias_axes='d',
				kernel_initializer=clone_initializer(self._kernel_initializer),
				bias_initializer=clone_initializer(self._bias_initializer),
				name='intermediate',
				**common_kwargs
		)
		policy = tf.keras.mixed_precision.global_policy()
		if policy.name == 'mixed_bfloat16': policy = tf.float32
		self._intermediate_activation_layer = tf.keras.layers.Activation(
			self._inner_activation, dtype=policy
		)
		self._inner_dropout_layer = tf.keras.layers.Dropout(
			rate=self._inner_dropout
		)
		self._output_dense = tf.keras.layers.EinsumDense(
			einsum_equation,
			output_shape=(None, last_output_shape),
			bias_axes='d',
			name='output',
			kernel_initializer=clone_initializer(self._kernel_initializer),
			bias_initializer=clone_initializer(self._bias_initializer),
			**common_kwargs
		)
		self._output_dropout = tf.keras.layers.Dropout(
			rate=self._output_dropout_rate
		)
		self._output_layer_norm = tf.keras.layers.LayerNormalization(
			name='output_layer_norm',
			axis=-1,
			epsilon=self._norm_epsilon,
			dtype=tf.float32
		)
		
		super().build(input_shape)
	
	def get_config(self):
		config = {
				'num_attention_heads': self._num_heads,
				'inner_dim': self._inner_dim,
				'inner_activation': self._inner_activation,
				'output_dropout': self._output_dropout_rate,
				'attention_dropout': self._attention_dropout_rate,
				'kernel_initializer': serialize_initializer(self._kernel_initializer, use_legacy_format=True),
				'bias_initializer': serialize_initializer(self._bias_initializer, use_legacy_format=True),
				'kernel_regularizer': serialize_regularizer(self._kernel_regularizer, use_legacy_format=True),
				'bias_regularizer': serialize_regularizer(self._bias_regularizer, use_legacy_format=True),
				'activity_regularizer': serialize_regularizer(self._activity_regularizer, use_legacy_format=True),
				'kernel_constraint': serialize_constraint(self._kernel_constraint, use_legacy_format=True),
				'bias_constraint': serialize_constraint(self._bias_constraint, use_legacy_format=True),
				'use_bias': self._use_bias,
				'norm_first': self._norm_first,
				'norm_epsilon': self._norm_epsilon,
				'inner_dropout': self._inner_dropout,
				'attention_initializer': serialize_initializer(self._attention_initializer, use_legacy_format=True),
				'attention_axes': self._attention_axes,
				'use_query_residual': self._use_query_residual,
				'key_dim': self._key_dim,
				'value_dim': self._value_dim,
				'output_last_dim': self._output_last_dim,
				'diff_q_kv_att_layer_norm': self._diff_q_kv_att_layer_norm,
		}
		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))
	
	def call(self, inputs, output_range=None):
		""" Transformer self-attention encoder block call.
			
			Args:
				inputs: a single tensor or a list of tensors. `input tensor` as the single
					sequence of embeddings. [`input tensor`, `attention mask`] to have the
					additional attention mask. [`query tensor`, `key value tensor`,
					`attention mask`] to have separate input streams for the query, and
					key/value to the multi-head attention.
				output_range: the sequence output range, [0, output_range) for slicing the
					target sequence. `None` means the target sequence is not sliced. If you
					would like to have no change to the model training, it is better to only
					set the `output_range` for serving.
			
			Returns:
				An output tensor with the same dimensions as input/query tensor.
		"""
		if isinstance(inputs, (list, tuple)):
			if len(inputs) == 2:
				input_tensor, attention_mask = inputs
				key_value = None
			elif len(inputs) == 3:
				input_tensor, key_value, attention_mask = inputs
			else:
				raise ValueError('Unexpected inputs to %s with length at %d' % (self.__class__, len(inputs)))
		else:
			input_tensor, key_value, attention_mask = (inputs, None, None)
		
		if output_range:
			if self._norm_first:
				source_tensor = input_tensor[:, 0:output_range, :]
				input_tensor = self._attention_layer_norm(input_tensor)
				if key_value is not None:
					key_value = self._attention_layer_norm_kv(key_value)
			target_tensor = input_tensor[:, 0:output_range, :]
			if attention_mask is not None:
				attention_mask = attention_mask[:, 0:output_range, :]
		else:
			if self._norm_first:
				source_tensor = input_tensor
				input_tensor = self._attention_layer_norm(input_tensor)
				if key_value is not None:
					key_value = self._attention_layer_norm_kv(key_value)
			target_tensor = input_tensor
		
		if key_value is None: key_value = input_tensor
		
		if self._return_attention_scores:
			attention_output, attention_scores = self._attention_layer(
				query=target_tensor,
				value=key_value,
				attention_mask=attention_mask,
				return_attention_scores=True
			)
		else:
			attention_output = self._attention_layer(
				query=target_tensor,
				value=key_value,
				attention_mask=attention_mask
			)
		attention_output = self._attention_dropout(attention_output)
		
		if self._norm_first:
			if self._use_query_residual:
				attention_output = source_tensor + attention_output
		else:
			if self._use_query_residual:
				attention_output = target_tensor + attention_output
			attention_output = self._attention_layer_norm(attention_output)
		
		if self._norm_first:
			source_attention_output = attention_output
			attention_output = self._output_layer_norm(attention_output)
		inner_output = self._intermediate_dense(attention_output)
		inner_output = self._intermediate_activation_layer(inner_output)
		inner_output = self._inner_dropout_layer(inner_output)
		layer_output = self._output_dense(inner_output)
		layer_output = self._output_dropout(layer_output)
		
		if self._norm_first:
			layer_output = source_attention_output + layer_output
		else:
			layer_output = tf.cast(layer_output, tf.float32)
			layer_output = self._output_layer_norm(layer_output + attention_output)
		
		if not self._return_sequences: layer_output = layer_output[:,-1,:]
		
		if self._return_attention_scores: return layer_output, attention_scores
		else: return layer_output
