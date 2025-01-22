''' Author: Dominik Beese
>>> Utility functions for model image resolution and batch sizes.
<<<
'''

MODEL_RESOLUTIONS = {
	'SimpleCNN': 224,
	'ResNet50': 224,
	'ResNet101': 224,
	'ResNet152': 224,
	'VGG16': 224,
	'VGG19': 224,
	'InceptionV3': 299,
	'EfficientNetB0': 224,
	'EfficientNetB1': 240,
	'EfficientNetB2': 260,
	'DenseNet121': 224,
	'DenseNet169': 224,
	'DenseNet201': 224,
	'MobileNet': 224,
	'NasNetMobile': 224,
	'NasNetLarge': 331,
}
MODEL_RESOLUTIONS = {k.lower(): v for k, v in MODEL_RESOLUTIONS.items()}

def parse(architecture: str) -> tuple[str, str]:
	parts = architecture.replace('-', '_').split('_')
	if len(parts) == 2: return tuple(parts)
	elif len(parts) == 1: return parts[0], None
	else: raise ValueError()

def sizeof(model_or_architecture: str) -> int:
	model_name = parse(model_or_architecture)[0]
	if model_name.lower() not in MODEL_RESOLUTIONS: raise NotImplementedError('Unknown model: %s' % model_name)
	return MODEL_RESOLUTIONS[model_name.lower()]
