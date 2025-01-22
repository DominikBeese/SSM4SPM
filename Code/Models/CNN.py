''' Author: Dominik Beese
>>> Implementation of several CNN models.
	- ResNet50 & ResNet101 & ResNet152 (https://arxiv.org/pdf/1512.03385)
	- VGG16 & VGG19 (https://arxiv.org/pdf/1409.1556)
	- InceptionV3 (https://arxiv.org/pdf/1512.00567)
	- EfficientNetB0 & EfficientNetB1 & EfficientNetB2 (https://arxiv.org/pdf/1905.11946)
	- DenseNet121 & DenseNet169 & DenseNet201 (https://arxiv.org/pdf/1608.06993)
	- MobileNet (https://arxiv.org/pdf/1704.04861)
	- NASNetMobile & NASNetLarge (https://arxiv.org/pdf/1707.07012)
<<<
'''

import tensorflow as tf


class CNN(tf.keras.Model):
	""" A CNN model.
		If possible, pre-trained on ImageNet.
	"""
	
	def __init__(
		self,
		cnn: str = 'InceptionV3',
		cnn_kwargs: dict = None,
		*args, **kwargs
	):
		""" Initializes the CNN model.
			cnn - the CNN architecture to use
			cnn_kwargs - additional kwargs for the CNN application
		"""
		super().__init__(*args, **kwargs)
		if cnn.lower() not in ['resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'inceptionv3', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'nasnetmobile', 'nasnetlarge']:
			raise NotImplementedError('CNN architecture not implemented: %s' % cnn)
		self.cnn = cnn
		self.cnn_kwargs = cnn_kwargs or dict()
	
	def build(self, input_shape):
		""" Builds the model.
			The input shape is in the format (None, image_width, image_height, image_channels).
		"""
		# CNN
		if self.cnn.lower() == 'resnet50': cnn_application = tf.keras.applications.ResNet50
		elif self.cnn.lower() == 'resnet101': cnn_application = tf.keras.applications.ResNet101
		elif self.cnn.lower() == 'resnet152': cnn_application = tf.keras.applications.ResNet152
		elif self.cnn.lower() == 'vgg16': cnn_application = tf.keras.applications.VGG16
		elif self.cnn.lower() == 'vgg19': cnn_application = tf.keras.applications.VGG19
		elif self.cnn.lower() == 'inceptionv3': cnn_application = tf.keras.applications.InceptionV3
		elif self.cnn.lower() == 'efficientnetb0': cnn_application = tf.keras.applications.EfficientNetB0
		elif self.cnn.lower() == 'efficientnetb1': cnn_application = tf.keras.applications.EfficientNetB1
		elif self.cnn.lower() == 'efficientnetb2': cnn_application = tf.keras.applications.EfficientNetB2
		elif self.cnn.lower() == 'densenet121': cnn_application = tf.keras.applications.DenseNet121
		elif self.cnn.lower() == 'densenet169': cnn_application = tf.keras.applications.DenseNet169
		elif self.cnn.lower() == 'densenet201': cnn_application = tf.keras.applications.DenseNet201
		elif self.cnn.lower() == 'mobilenet': cnn_application = tf.keras.applications.MobileNet
		elif self.cnn.lower() == 'nasnetmobile': cnn_application = tf.keras.applications.NASNetMobile
		elif self.cnn.lower() == 'nasnetlarge': cnn_application = tf.keras.applications.NASNetLarge
		else: raise Exception()
		self.model = cnn_application(
			include_top=True,
			weights='imagenet', # pre-trained on ImageNet
			classifier_activation=None, # no softmax output
			**self.cnn_kwargs,
		)
		self.model.build(input_shape=input_shape[1:]) # IN: (image_width, image_height, image_channels), OUT: (features,)
	
	def call(self, inputs, **kwargs):
		""" Calls the model.
			Note: training=False, cf. https://www.tensorflow.org/guide/keras/transfer_learning#fine-tuning
		"""
		# Input Preprocessing
		if self.cnn.lower() in ['resnet50', 'resnet101', 'resnet152']: cnn_application = tf.keras.applications.resnet
		elif self.cnn.lower() == 'vgg16': cnn_application = tf.keras.applications.vgg16
		elif self.cnn.lower() == 'vgg19': cnn_application = tf.keras.applications.vgg19
		elif self.cnn.lower() == 'inceptionv3': cnn_application = tf.keras.applications.inception_v3
		elif self.cnn.lower() in ['efficientnetb0', 'efficientnetb1', 'efficientnetb2']: cnn_application = tf.keras.applications.efficientnet
		elif self.cnn.lower() in ['densenet121', 'densenet169', 'densenet201']: cnn_application = tf.keras.applications.densenet
		elif self.cnn.lower() == 'mobilenet': cnn_application = tf.keras.applications.mobilenet
		elif self.cnn.lower() in ['nasnetmobile', 'nasnetlarge']: cnn_application = tf.keras.applications.nasnet
		else: raise Exception()
		inputs = cnn_application.preprocess_input(tf.cast(inputs, tf.keras.backend.floatx()))
		
		# CNN
		return self.model(inputs, training=False)
