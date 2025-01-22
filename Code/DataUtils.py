''' Author: Dominik Beese
>>> Utility functions for data management.
<<<
'''

from os.path import join
from tqdm.auto import tqdm
import json

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import tensorflow as tf

from typing import Union


### HELPER ###

def _get_frames(frames: Union[int, list[Union[int, float]]]) -> list[int]:
	""" Returns the individual frames given the frame specification.
		1-value specification - specifies the individual frame
		2-tuple specification - specifies the minimum (inclusive) and maximum (exclusive) frames
		3-tuple specification - specifies the minimum (inclusive) and maximum (exclusive) frames, plus the spacing (float) of the frames
		Otherwise, it's the individual frame specification and all individual frames are returned.
	"""
	if isinstance(frames, int):
		return frames
	if len(frames) == 2:
		return range(frames[0], frames[1])
	elif len(frames) == 3:
		i, r = 0, list()
		while True:
			f = int(i * frames[2] + frames[0])
			if f >= frames[1]: break
			r.append(f)
			i += 1
		return r
	return frames

def _random_frames(start: int, end: int, count: int) -> list[int]:
	return sorted(np.random.permutation(end - start)[:count] + start)

def _get_labels(label: Union[int, list[int]]):
	if isinstance(label, pd.Series):
		if isinstance(label.get(0), list):
			if any(isinstance(e, list) for e in label.get(0)):
				return tuple(np.array([l[i] for l in label]) for i in range(len(label.get(0))))
			else:
				return tuple([np.array([np.array(l) for l in label])])
		else:
			return tuple([np.array(label)])
	return label

def _tqdm(iterable, progress_bar: bool = True, **kwargs):
	if progress_bar: return tqdm(iterable, **kwargs)
	return iterable

def _load(
	caseId: int,
	frames: Union[int, list[int]],
	data_path: str,
	target_width: int = 224,
	target_height: int = 224,
	image_format: str = 'jpg',
	image_dtype: tf.DType = tf.uint8,
	model_name: str = None,
	features_format: str = 'npy',
):
	""" Loads the images/features from the given caseId from the given data path as a tensorflow tensor.
		If model_name is given, loads the features.
		Otherwise, loads the images in the given dimensions.
		Note: The images from the videos are 1-indexed, the frames in the annotations are 0-indexed.
	"""
	if model_name is None:
		return tf.cast(
			tf.image.resize(
				tf.convert_to_tensor(
					np.array(
						[tf.keras.utils.load_img(join(data_path, '%04d' % caseId, '%05d.%s' % (frame + 1, image_format))) for frame in frames]
						if hasattr(frames, '__iter__') else
						tf.keras.utils.load_img(join(data_path, '%04d' % caseId, '%05d.%s' % (frames + 1, image_format)))
					)
				), (target_width, target_height)
			), dtype=image_dtype
		)
	else:
		return tf.convert_to_tensor(
			np.load(join(data_path, model_name, '%04d.%s' % (caseId, features_format)))[frames]
		)


### LOADING - EXPERIMENTS FROM SPLITS ###

def load_from_splits(
	dataset: pd.DataFrame,
	data_path: str,
	image_size: int = None,
	image_width: int = 224,
	image_height: int = 224,
	model_name: str = None,
	feature_dimension: int = 1000,
	quick: bool = False,
	progress_bar: bool = True,
	dtype: tf.DType = None,
	label_dtype: tf.DType = tf.int32,
):
	""" Loads the images/features for the given dataset from the given data path.
		Returns the encoded dataset, incl. labels if the dataset contains labels.
		- dataset: the splits as a pandas DataFrame (w/ columns: caseId, frames, opt. label)
		- data_path: the path to all images and the manifest file
		- image_size: if given, used for image_width and image_height
		- image_width & image_height: target dimensions of the images to load
		- model_name: if given, ignores image_{size,width,height} and loads pre-computed features instead
		- feature_dimension: if quick loading is True and mode_name is given, this is the dimensionality of the pre-computed features
		- quick: loading type (see below)
		- progress_bar: whether to show a progress bar
		- dtype: the data type of the image or features; if None, automatically inferred
		- label_dtype: the data type of the labels
		Quick loading has a memory limit.
		Non-quick loading only supports loading a fixed number of frames for each clip.
	"""
	# arguments
	if image_size is not None: image_width, image_height = image_size, image_size
	if dtype is None: dtype = tf.uint8 if model_name is None else tf.float16
	
	# load manifest
	with open(join(data_path, 'manifest.json'), 'r', encoding='UTF-8') as file:
		manifest = json.load(file)
	
	# load clips
	with_labels = 'label' in dataset
	if quick:
		sequence = list()
		for _, clip in _tqdm(dataset.iterrows(), total=len(dataset), progress_bar=progress_bar):
			sequence.append(_load(clip['caseId'], _get_frames(clip['frames']), data_path, target_width=image_width, target_height=image_height, image_format=manifest['image-format'], model_name=model_name, image_dtype=dtype))
		if with_labels:
			labels = _get_labels(dataset['label'])
			dataset1 = tf.data.Dataset.from_tensor_slices(sequence)
			dataset2 = tf.data.Dataset.from_tensor_slices(labels)
			return tf.data.Dataset.zip((dataset1, dataset2)).prefetch(1)
		else:
			return tf.data.Dataset.from_tensor_slices(sequence).prefetch(1)
	else:
		if dataset['frames'].map(type).unique()[0] == list and dataset['frames'].map(len).nunique() > 1: raise Exception('No unique number of frames. Quick loading not supported.')
		frame_count = len(_get_frames(dataset.iloc[0]['frames'])) if isinstance(dataset.iloc[0]['frames'], list) else 1
		if with_labels: labels = _get_labels(dataset['label'])
		def _loader():
			for i, (_, clip) in enumerate(_tqdm(dataset.iterrows(), total=len(dataset), progress_bar=progress_bar)):
				image = _load(clip['caseId'], _get_frames(clip['frames']), data_path, target_width=image_width, target_height=image_height, image_format=manifest['image-format'], model_name=model_name, image_dtype=dtype)
				if with_labels: yield image, tuple(label[i] for label in labels)
				else: yield image
		if frame_count > 1: image_shape = (frame_count, image_width, image_height, 3) if model_name is None else (frame_count, feature_dimension)
		else: image_shape = (image_width, image_height, 3) if model_name is None else (feature_dimension,)
		if with_labels: output_signature = (tf.TensorSpec(shape=image_shape, dtype=dtype), tuple(tf.TensorSpec(shape=label.shape[1:], dtype=label_dtype) for label in labels))
		else: output_signature = tf.TensorSpec(shape=image_shape, dtype=dtype)
		return tf.data.Dataset.from_generator(_loader, output_signature=output_signature).prefetch(1)


### LOADING - EXPERIMENTS FROM DATA ###

def load_clips(
	data_path: str,
	frame_count: int = 10,
	image_size: int = None,
	image_width: int = 224,
	image_height: int = 224,
	model_name: str = None,
	clip_offset: float = 0.0,
	clip_length: float = 3.0,
	clip_overlap: float = 1.0,
	max_clips: int = -1,
	label_extractor: callable = None,
	num_labels: int = -1,
	quick: bool = False,
	progress_bar: bool = True,
	dtype: tf.DType = None,
	label_dtype: tf.DType = tf.int32,
):
	""" Loads the images from the given data path according to the desired clip offset, length and overlap.
		Returns the dataset and the encoded dataset, incl. labels if a label extractor is given.
		- data_path: the path to all images and the manifest file
		- frame_count: the number of frames randomly sampled from each clip
		- image_size: if given, used for image_width and image_height
		- image_width & image_height: target dimensions of the images to load
		- model_name: if given, ignores image_{size,width,height} and loads pre-computed features instead
		- clip_offset: the clip offset in seconds
		- clip_length: the clip length in seconds
		- clip_overlap: the clip overlap in seconds
		- max_clips: loads at most this many clips
		- label_extractor: [deprecated] a function to extract the label from the given metadata of the phase
		- num_labels: [deprecated] the number of labels (must be given if quick is False)
		- quick: loading type (see below)
		- progress_bar: whether to show a progress bar
		- dtype: the data type of the image or features; if None, automatically inferred
		- label_dtype: the data type of the labels
		Quick loading has a memory limit.
		Non-quick loading only supports loading a fixed number of frames for each clip.
	"""
	# deprecated arguments
	if label_extractor is not None: raise DeprecationWarning()
	if num_labels != -1: raise DeprecationWarning()
	
	# arguments
	if quick is False and label_extractor is not None and num_labels <= 0:
		raise Exception('A label count must be given if quick loading is False')
	if num_labels <= 0: num_labels = None
	if image_size is not None: image_width, image_height = image_size, image_size
	if dtype is None: dtype = tf.uint8 if model_name is None else tf.float16
	
	# load manifest and phases
	with open(join(data_path, 'manifest.json'), 'r', encoding='UTF-8') as file:
		manifest = json.load(file)
	with open(join(data_path, 'phases.json'), 'r', encoding='UTF-8') as file:
		phases = json.load(file)
	
	# seconds to frames
	clip_length = int(clip_length * manifest['fps'])
	clip_overlap = int(clip_overlap * manifest['fps'])
	clip_offset = int(clip_offset * manifest['fps'])
	
	# split into clips
	with_labels = label_extractor is not None
	dataset = pd.DataFrame(columns=['caseId', 'frames', 'label'] if with_labels else ['caseId', 'frames'])
	def _loader():
		nonlocal dataset
		n = 0
		dataset.drop(dataset.index, inplace=True)
		for phase in _tqdm(phases, progress_bar=progress_bar):
			length = phase['end'] - phase['start'] + 1
			for s in range(clip_offset, length - clip_length, clip_length - clip_overlap):
				start, end = phase['start'] + s, phase['start'] + s + clip_length
				images = _load(phase['caseId'], _random_frames(start, end, frame_count), data_path, target_width=image_width, target_height=image_height, image_format=manifest['image-format'], model_name=model_name, image_dtype=dtype)
				if with_labels:
					label = label_extractor(phase)
					dataset.loc[n] = (phase['caseId'], (start, end), label)
					if quick:
						yield images
					else:
						label = _get_labels(label, num_classes=num_labels)
						yield images, label
				else:
					dataset.loc[n] = (phase['caseId'], (start, end))
					yield images
				n += 1
				if max_clips > 0 and n >= max_clips: return
	
	# load clips
	if quick:
		images = list(_loader())
		if with_labels:
			labels = _get_labels(dataset['label'])
			return dataset, tf.data.Dataset.from_tensor_slices((images, labels)).prefetch(1)
		else:
			return dataset, tf.data.Dataset.from_tensor_slices(images).prefetch(1)
	else:
		if with_labels:
			output_signature = (tf.TensorSpec(shape=(clip_length, image_width, image_height, 3), dtype=dtype), tf.TensorSpec(shape=None, dtype=label_dtype))
		else:
			output_signature = tf.TensorSpec(shape=(clip_length, image_width, image_height, 3), dtype=dtype)
		return dataset, tf.data.Dataset.from_generator(_loader, output_signature=output_signature).prefetch(1)

def load_videos(
	data_path: str,
	case_ids: list[int],
	image_size: int = None,
	image_width: int = 224,
	image_height: int = 224,
	model_name: str = None,
	sample_fps: float = 10/3,
	sample_length: int = 3,
	sample_count: int = 1,
	clip_offset_mult: int = 5,
	clip_offset_add: int = 0,
	max_clips: int = -1,
	quick: bool = False,
	progress_bar: bool = True,
	dtype: tf.DType = None,
):
	""" Loads the images from the given data path according to the desired sample fps, sample length and clip offsets.
		Returns the dataset and the encoded dataset, without labels.
		- data_path: the path to all images and the manifest file
		- case_ids: the list of case ids of the videos to load
		- image_size: if given, used for image_width and image_height
		- image_width & image_height: target dimensions of the images to load
		- model_name: if given, ignores image_{size,width,height} and loads pre-computed features instead
		- sample_fps: the fixed number frames per second to extract the video at
		- sample_length: the length of each extracted video clip in seconds
		- sample_count: the number of samples per video clip
		- clip_offset_mult: the offset of individual clips as multiples of the inter-frame distance, i.e. video_fps / sample_fps
		- clip_offset_add: the additive offset of individual clips as an absolute number of frames
		- max_clips: loads at most this many clips
		- quick: loading type (see below)
		- progress_bar: whether to show a progress bar
		- dtype: the data type of the image or features; if None, automatically inferred
		Quick loading has a memory limit.
		Non-quick loading only supports loading a fixed number of frames for each clip.
	"""
	# arguments
	if image_size is not None: image_width, image_height = image_size, image_size
	if dtype is None: dtype = tf.uint8 if model_name is None else tf.float16
	
	# load manifest and cases
	with open(join(data_path, 'manifest.json'), 'r', encoding='UTF-8') as file:
		manifest = json.load(file)
	with open(join(data_path, 'cases.json'), 'r', encoding='UTF-8') as file:
		cases = json.load(file)
	cases = {c['caseId']: c for c in cases}
	
	# seconds to frames
	sample_frames = int(manifest['fps'] / sample_fps)
	clip_length = int(sample_length * manifest['fps'])
	
	# split into clips
	dataset = pd.DataFrame(columns=['caseId', 'frames'])
	def _loader():
		nonlocal dataset
		n = 0
		dataset.drop(dataset.index, inplace=True)
		for case_id in _tqdm(case_ids, progress_bar=progress_bar):
			length = cases[case_id]['frames']
			for sample in range(sample_count):
				sample_offset = sample * int(manifest['fps'] / sample_fps / sample_count)
				for s in range(sample_offset, length - clip_length + sample_frames, sample_frames * clip_offset_mult + clip_offset_add):
					frames = [s, s + clip_length, sample_frames]
					images = _load(case_id, _get_frames(frames), data_path, target_width=image_width, target_height=image_height, image_format=manifest['image-format'], model_name=model_name, image_dtype=dtype)
					dataset.loc[n] = (case_id, tuple(frames))
					yield images
					n += 1
					if max_clips > 0 and n >= max_clips: return
	
	# load clips
	if quick:
		images = list(_loader())
		return dataset, tf.data.Dataset.from_tensor_slices(images).prefetch(1)
	else:
		output_signature = tf.TensorSpec(shape=(clip_length, image_width, image_height, 3), dtype=dtype)
		return dataset, tf.data.Dataset.from_generator(_loader, output_signature=output_signature).prefetch(1)


### LOADING - INDIVIDUAL VIDEOS ###

def load_video(
	caseId: int,
	frames: int,
	batch_size: int,
	data_path: str,
	image_size: int = None,
	image_width: int = 224,
	image_height: int = 224,
	dtype: tf.DType = tf.uint8,
):
	""" Loads all frames from the given video.
		Returns the encoded dataset.
		- caseId: the id of the video to load
		- frames: the total number of frames of the given video
		- data_path: the path to all images and the manifest file
		- image_size: if given, used for image_width and image_height
		- image_width & image_height: target dimensions of the images to load
		- dtype: the data type of the image
		Non-quick loading by default.
	"""
	# arguments
	if image_size is not None: image_width, image_height = image_size, image_size
	
	# load manifest
	with open(join(data_path, 'manifest.json'), 'r', encoding='UTF-8') as file:
		manifest = json.load(file)
	
	# load video
	def _loader():
		for frame in range(0, frames, batch_size):
			yield _load(caseId, range(frame, min(frame+batch_size, frames)), data_path, target_width=image_width, target_height=image_height, image_format=manifest['image-format'], image_dtype=dtype)
	output_signature = tf.TensorSpec(shape=(None, image_width, image_height, 3), dtype=dtype)
	return tf.data.Dataset.from_generator(_loader, output_signature=output_signature)


### EXTRACTION ###

def get_image(dataset: tf.data.Dataset, i: int = 0, j: int = 0):
	""" Returns the j-th frame from the i-th clip in the given dataset. """
	iterator = dataset.as_numpy_iterator()
	while i > 0: next(iterator); i -= 1
	elem = next(iterator)
	if len(dataset._structure) > 1: elem = elem[0]
	return elem[j,:]

def get_label(dataset: tf.data.Dataset, i: int = 0):
	""" Returns the annotation for the i-th clip in the given dataset. """
	if len(dataset._structure) == 1: raise Exception('Dataset has no annotations.')
	iterator = dataset.as_numpy_iterator()
	while i > 0: next(iterator); i -= 1
	return next(iterator)[1]
