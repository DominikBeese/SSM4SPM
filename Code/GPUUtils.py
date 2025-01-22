''' Author: Dominik Beese
>>> Utility functions for managing gpus.
<<<
'''

from typing import Union

import tensorflow as tf

def set_visible_devices(devices: Union[list[int], bool]) -> list[str]:
	if devices is True:
		pass
	elif devices is False:
		tf.config.set_visible_devices(list(), 'GPU')
	else:
		physical_devices = tf.config.list_physical_devices('GPU')
		tf.config.set_visible_devices([physical_devices[k] for k in devices], 'GPU')
	logical_devices = tf.config.list_logical_devices('GPU')
	print('Tensorflow is using:', [physical_devices[k].name for k in devices] if logical_devices else 'CPU')
	return [tf.config.experimental.get_device_details(physical_devices[k])['device_name'] for k in devices]
