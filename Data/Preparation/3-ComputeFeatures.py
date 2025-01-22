''' Author: Dominik Beese
>>> Use pre-trained CNNs to pre-compute features for all videos.
<<<
'''

import sys
from os.path import join, exists
from os import makedirs
from tqdm.auto import tqdm
import json

import pandas as pd
import numpy as np


### CONFIGURATION ###

MODELS = ['DenseNet121', 'ResNet152', 'MobileNet', 'ResNet50', 'VGG16', 'InceptionV3', 'EfficientNetB0', 'DenseNet121', 'ResNet101', 'ResNet152', 'VGG19', 'EfficientNetB1', 'EfficientNetB2', 'DenseNet169', 'DenseNet201']
BATCH_SIZE = 64

sys.path.append(join('..', '..', 'Code'))
from Models import CNN, ModelUtils
import DataUtils


### DO IT ###

data_path = r'..'
target_path = r'..'

with open(join(data_path, 'cases.json'), 'r', encoding='UTF-8') as file:
	cases = json.load(file)

for cnn in tqdm(MODELS, desc='Models'):
	model = None
	model_path = join(target_path, cnn)
	makedirs(join(model_path), exist_ok=True)
	for case in tqdm(cases, desc='Cases', leave=False):
		output_file = join(model_path, '%d.npy' % case['caseId'])
		if exists(output_file): continue
		if model is None: model = CNN(cnn=cnn)
		encoded_dataset = DataUtils.load_video(case['caseId'], case['frames'], BATCH_SIZE, data_path, ModelUtils.sizeof(cnn))
		features = model.predict(encoded_dataset, verbose=None)
		np.save(output_file, features.astype('float16'))
