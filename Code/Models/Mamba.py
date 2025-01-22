''' Author: Dominik Beese
>>> Implementation of a Mamba Block and Mamba Model by Vedant Jumle, edited by me.
	* Mamba (https://arxiv.org/pdf/2312.00752)
	* Implementation (https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-tf.keras-and-tensorflow-32d6d4b32546)
<<<
'''

import numpy as np
import tensorflow as tf

from dataclasses import dataclass
from einops import rearrange, repeat

from typing import Union


### MAMBA BLOCK ###

@dataclass
class MambaArgs:
	""" Parameters for the Mamba block.
		d_model - model dimension, i.e. D in the paper, e.g. 768, 1024, 1536, 2048, 2560
		e_model - block expansion factor, i.e. E in the paper, typically 2
		d_state - SSM state expansion factor / latent state dimension, i.e. N in the paper, typically 1, 2, 4, 8, 16, 32, 64, 128
		d_conv - local convolution width, typically 4
		conv_use_bias - whether to add a bias to the convolutional layer
		dense_use_bias - whether to add a bias to the dense layer
	"""
	d_model: int
	e_model: int = 2
	d_state: int = 16
	d_conv: int = 4
	conv_use_bias: bool = True
	dense_use_bias: bool = False
	
	def kwargs(self) -> dict:
		""" Returns the MambaArgs as a kwargs dict. """
		return self.__dict__
	
	@classmethod
	def split(cls, **kwargs) -> tuple[dict, dict]:
		""" Returns the MambaArgs and the remaining, non-MambaArgs, kwargs as a tuple of dicts. """
		keys = MambaArgs.__annotations__.keys()
		return {k: v for k, v in kwargs.items() if k in keys}, {k: v for k, v in kwargs.items() if k not in keys}
	
	def update(self, **kwargs) -> dict:
		""" Updates the given MambaArgs acording to the given kwargs.
			Returns the remaining, non-MambaArgs, kwargs.
		"""
		mamba_kwargs, kwargs = MambaArgs.split(**kwargs)
		for k, v in mamba_kwargs.items(): setattr(self, k, v)
		return kwargs

class MambaBlock(tf.keras.layers.Layer):
	""" Implementation of a single Mamba Block.
		Original paper: https://arxiv.org/pdf/2312.00752
	"""
	
	def __init__(self, mamba_args: MambaArgs = None, return_sequences: bool = True, *args, **kwargs):
		""" Initializes the Mamba block.
			mamba_args - configurations for the Mamba block
			return_sequences - whether to return the last output in the output sequence, or the full sequence
			*args - additional args for the keras layer
			**kwargs - overwrite values for the mamba_args, plus additional kwargs for the keras layer
		"""
		if mamba_args is None: mamba_args = MambaArgs(1)
		kwargs = mamba_args.update(**kwargs)
		super().__init__(*args, **kwargs)
		
		self.return_sequences = return_sequences
		
		self.model_internal_dim: int = int(mamba_args.e_model * mamba_args.d_model) # E*D = ED in the paper
		self.delta_t_rank = int(np.ceil(mamba_args.d_model / 16))
		
		self.in_projection = tf.keras.layers.Dense(
			self.model_internal_dim * 2,
			input_shape=(mamba_args.d_model,),
			use_bias=False
		)
		
		self.conv1d = tf.keras.layers.Conv1D(
			filters=self.model_internal_dim,
			use_bias=mamba_args.conv_use_bias,
			kernel_size=mamba_args.d_conv,
			groups=self.model_internal_dim,
			data_format='channels_first',
			padding='causal'
		)
		
		# this layer takes in current token 'x' and outputs the input-specific Δ, B, C (according to S6)
		self.x_projection = tf.keras.layers.Dense(
			self.delta_t_rank + mamba_args.d_state * 2,
			use_bias=False
		)
		
		# this layer projects Δ from delta_t_rank to the mamba internal dimension
		self.delta_t_projection = tf.keras.layers.Dense(self.model_internal_dim, input_shape=(self.delta_t_rank,), use_bias=True)
		
		self.A = tf.Variable(repeat(
			tf.range(1, mamba_args.d_state + 1, dtype=tf.float32),
			'n -> d n',
			d=self.model_internal_dim
		), trainable=False, dtype=tf.float32)
		self.A_log = tf.Variable(tf.math.log(self.A), trainable=False, dtype=tf.float32)
		
		self.D = tf.Variable(np.ones(self.model_internal_dim), dtype=tf.float32)
		
		self.out_projection = tf.keras.layers.Dense(
			mamba_args.d_model,
			input_shape=(self.model_internal_dim,),
			use_bias=mamba_args.dense_use_bias
		)
	
	def selective_scan(self, u, delta, A, B, C, D):
		dA = tf.einsum('bld,dn->bldn', delta, A) # first step of A_bar = exp(ΔA), i.e., ΔA
		dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)
		
		dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]
		
		dA_cumsum = tf.reverse(dA_cumsum, axis=[1]) # flip along axis 1
		
		# cumulative sum along all the input tokens, parallel prefix sum, calculates dA for all the input tokens parallely
		dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
		dA_cumsum = tf.exp(dA_cumsum) # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
		
		dA_cumsum = tf.reverse(dA_cumsum, axis=[1]) # flip back along axis 1
		
		x = dB_u * dA_cumsum
		x = tf.math.cumsum(x, axis=1)/(dA_cumsum + 1e-12) # 1e-12 to avoid division by 0
		
		y = tf.einsum('bldn,bln->bld', x, C)
		return y + u * D
	
	def ssm(self, x):
		""" Runs the SSM. See:
			- Algorithm 2 in Section 3.2 in the Mamba paper [https://arxiv.org/pdf/2312.00752]
			- run_SSM(A, B, C, u) in The Annotated S4 [https://github.com/state-spaces/mamba]
			
			Official Implementation:
				mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
		"""
		
		(d_in, n) = self.A_log.shape
		
		# Compute ∆ A B C D, the state space parameters.
		#	 A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
		#	 ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
		#								  and is why Mamba is called **selective** state spaces)
		
		A = -tf.exp(tf.cast(self.A_log, tf.float32)) # shape -> (d_in, n)
		D = tf.cast(self.D, tf.float32)
		
		x_dbl = self.x_projection(x) # shape -> (batch, seq_len, delta_t_rank + 2*n)
		
		delta, B, C = tf.split(
			x_dbl, 
			num_or_size_splits=[self.delta_t_rank, n, n],
			axis=-1
		) # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)
		
		delta = tf.nn.softplus(self.delta_t_projection( delta)) # shape -> (batch, seq_len, model_input_dim)
		return self.selective_scan(x, delta, A, B, C, D)
	
	def call(self, x):
		""" Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [https://arxiv.org/pdf/2312.00752].
			
			Official Implementation:
				class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
				mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
		"""
		
		batch_size, seq_len, dimension = x.shape
		
		x_and_res = self.in_projection(x) # shape = (batch, seq_len, 2 * model_internal_dimension)
		x, res = tf.split(x_and_res, [self.model_internal_dim, self.model_internal_dim], axis=-1)
		
		x = rearrange(x, 'b l d_in -> b d_in l')
		x = self.conv1d(x)[:, :, :seq_len]
		x = rearrange(x, 'b d_in l -> b l d_in')
		
		x = tf.nn.swish(x)
		y = self.ssm(x)
		y = y * tf.nn.swish(res) # right side of mamba block image
		
		outputs = self.out_projection(y)
		
		if self.return_sequences: return outputs
		else: return outputs[:,-1,:]

class ResidualBlock(tf.keras.layers.Layer):
	""" Implementation of a mamba block and a layer normalization. """
	
	def __init__(self, mamba_args: MambaArgs = None, return_sequences: bool = True, *args, **kwargs):
		""" Initializes a Mamba block and a layer normalization.
			mamba_args - configurations for the Mamba block
			return_sequences - whether to return the last output in the output sequence, or the full sequence
			*args - additional args for the keras layer
			**kwargs - overwrite values for the mamba_args, plus additional kwargs for the keras layer
		"""
		if mamba_args is None: mamba_args = MambaArgs(1)
		kwargs = mamba_args.update(**kwargs)
		super().__init__(*args, **kwargs)
		
		self.return_sequences = return_sequences
		
		self.mixer = MambaBlock(mamba_args=mamba_args, return_sequences=True)
		self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
	
	def call(self, x):
		"""
			Official Implementation:
				Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
				
				Note: the official repo chains residual blocks that look like
					[Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
				where the first Add is a no-op. This is purely for performance reasons as this
				allows them to fuse the Add->Norm.

				We instead implement our blocks as the more familiar, simpler, and numerically equivalent
					[Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
		"""
		outputs = self.mixer(self.norm(x)) + x
		
		if self.return_sequences: return outputs
		else: return outputs[:,-1,:]


### MAMBA MODEL ###

class MambaModel(tf.keras.Model):
	""" Implementation of a simple Mamba model. """
	
	def __init__(
		self,
		n_layers: int,
		mamba_args: MambaArgs = None,
		num_labels: int = 2,
		dropout_rate: float = 0.2,
		embedding_vocab_size: int = None,
		embedding_sequence_length: int = None,
		use_lm_head: bool = False,
		*args,
		**kwargs
	):
		""" Initializes the Mamba model.
			mamba_args - configurations for the Mamba block
			n_layers - number of Mamba blocks/layers, e.g. 24, 48, 64
			num_labels - number of output nodes
			dropout_rate - dropout rate for the dropout layer after each residual block
			embedding_vocab_size - if given, adds an input and embedding layer with the given vocab size
			embedding_sequence_length - if given, adds an input and embedding layer with the given input length
			use_lm_head - whether to use a sequence output
			*args - additional args for the keras layer
			**kwargs - overwrite values for the mamba_args, plus additional kwargs for the keras layer
		"""
		if use_lm_head and embedding_sequence_length is not None: num_classes = embedding_vocab_size
		if use_lm_head: output_activation = None
		else: output_activation = 'sigmoid' if num_labels == 1 else 'softmax'
		if (embedding_vocab_size is None) != (embedding_sequence_length is None): raise ValueError()
		if mamba_args is None: mamba_args = MambaArgs(1)
		kwargs = mamba_args.update(**kwargs)
		
		if embedding_sequence_length is not None:
			input_layer = tf.keras.layers.Input(shape=(embedding_sequence_length,))
			x = tf.keras.layers.Embedding(embedding_vocab_size, mamba_args.d_model, input_length=embedding_sequence_length)(input_layer)
		else: raise NotImplementedError()
		
		for i in range(n_layers):
			x = ResidualBlock(mamba_args=mamba_args, name='Residual_%d' % i)(x)
			x = tf.keras.layers.Dropout(dropout_rate)(x)
		
		x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
		
		if not use_lm_head: x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024, activation=tf.nn.gelu)(x)
		output_layer = tf.keras.layers.Dense(num_labels, activation=output_activation)(x)
		
		super().__init__(inputs=input_layer, outputs=output_layer, name='Mamba', *args, **kwargs)


### EXAMPLE ###

if __name__ == '__main__':
	
	import tensorflow_addons as tfa
	from transformers import AutoTokenizer
	from datasets import load_dataset
	from tqdm import tqdm
	
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	vocab_size = tokenizer.vocab_size
	
	SEQ_LENGTH = 32 # 128
	
	model = MambaModel(
		mamba_args=MambaArgs(d_model=16, d_states=32), # MambaArgs(d_model=128, d_states=32),
		n_layers=5, # 12,
		num_labels=1,
		embedding_vocab_size=vocab_size,
		embedding_sequence_length=SEQ_LENGTH,
	)
	model.compile(
		loss=['binary_crossentropy'],
		optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004),
		metrics=['accuracy'],
	)
	model.summary()
	
	dataset = load_dataset('ajaykarthick/imdb-movie-reviews')
	
	train_labels, test_labels = [], []
	train_ids = np.zeros((len(dataset['train']), SEQ_LENGTH))
	test_ids = np.zeros((len(dataset['test']), SEQ_LENGTH))
	
	for i, item in enumerate(tqdm(dataset['train'])):
		text = item['review']
		train_ids[i, :] = tokenizer.encode_plus(text, max_length=SEQ_LENGTH, padding='max_length', return_tensors='np')['input_ids'][0][:SEQ_LENGTH]
		train_labels.append(item['label'])
	
	for i, item in enumerate(tqdm(dataset['test'])):
		text = item['review']
		test_ids[i, :] = tokenizer.encode_plus(text, max_length=SEQ_LENGTH, padding='max_length', return_tensors='np')['input_ids'][0][:SEQ_LENGTH]
		test_labels.append(item['label'])
	del dataset
	
	BATCH_SIZE = 32
	train_dataset = tf.data.Dataset.from_tensor_slices((train_ids, train_labels)).batch(BATCH_SIZE).shuffle(1000)
	test_dataset = tf.data.Dataset.from_tensor_slices((test_ids, test_labels)).batch(BATCH_SIZE).shuffle(1000)
	
	history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)
	
	for text in ['This movie is amazing.', 'This movie is quite okay.', 'The movie sucks.']:
		text = '[CLS] ' + text
		tokens = tokenizer.encode(text, max_length=SEQ_LENGTH, padding='max_length', return_tensors='np')
		score = model(tokens)[0, 0].numpy()
		print(text, score)
