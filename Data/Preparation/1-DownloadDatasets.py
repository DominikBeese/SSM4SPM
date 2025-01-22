''' Author: Dominik Beese
>>> Download All Cataract-1K Datasets (https://www.synapse.org/#!Synapse:syn52540135/wiki/626061)
<<<
'''

from urllib.request import urlretrieve
from shutil import rmtree, move
from os import remove
from os.path import join, exists
from zipfile import ZipFile
from tqdm.auto import tqdm
import json


### CONFIGURATION ###

synapse_token = r'' # enter your synapse token

### HELPER ###

def titleify(title, n=80):
	print('='*n)
	p = (n-len(title))//2-1
	q = n-len(title)-p-2
	print('='*p, title, '='*q)
	print('='*n)
	return join('..', title + '-raw')

class tqdm_hook:
	def __init__(self):
		self.t = None
		self.j = 0
	def __call__(self, i, k, n):
		if self.t is None: self.t = tqdm(total=-(-n//k))
		else: self.t.update(i - self.j)
		self.j = i


### SETUP ###

def install_and_import(package):
	import importlib, subprocess, sys
	try: importlib.import_module(package)
	except ImportError: subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
	finally: globals()[package] = importlib.import_module(package)

def synapse_download(entity, name):
	if not exists(p := join(path, name)):
		synapseutils.syncFromSynapse(syn, entity, path=p)

# Cataract-1K
path = titleify('Cataract-1K')

install_and_import('synapseclient')
install_and_import('synapseutils')

syn = synapseclient.Synapse()
syn.login(authToken=synapse_token)
#synapse_download('syn53404507', 'Cataract_1K_videos') # no annotations
synapse_download('syn53395146', 'Phase_recognition_dataset')
synapse_download('syn53395479', 'Segmentation_dataset')
synapse_download('syn53395131', 'Lens_irregularity')
synapse_download('syn53395402', 'Pupil_reaction')

filename = join(path, 'Cataract-1K.zip')
if not exists(filename):
	urlretrieve(r'https://codeload.github.com/Negin-Ghamsarian/Cataract-1K/zip/refs/heads/main', filename=filename, reporthook=tqdm_hook())
ZipFile(filename).extractall(path)
remove(filename)
for folder in ['Dataset_codes', 'TrainIDs_Cataract_1k_Anatomy_Instruments', 'TrainIDs_Cataract_1k_Instruments']:
	move(join(path, 'Cataract-1K-main', folder), join(path, folder))
rmtree(join(path, 'Cataract-1K-main'))
