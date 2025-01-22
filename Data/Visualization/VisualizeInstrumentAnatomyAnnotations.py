''' Author: Dominik Beese
>>> Visualize instrument and anatomy annotations for each video
<<<
'''

from os.path import join
import json
from tqdm.auto import tqdm
import itertools

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches

import sys
sys.path.append('..')
from InstrumentAnatomyRecognitionDataset import Helper


### CONFIGURATION ###

GROUPS = [['Iris'], ['Pupil'], ['Intraocular Lens', 'Slit/Incision Knife'], ['Gauge', 'Spatula'], ['Capsulorhexis Cystotome', 'Phacoemulsification Tip', 'Irrigation-Aspiration', 'Lens Injector'], ['Capsulorhexis Forceps', 'Katena Forceps']]

with open(join('..', 'manifest.json'), 'r', encoding='UTF-8') as file:
	manifest = json.load(file)


### COMPUTE IT ###

cs = pd.read_json(join('..', 'cases.json')).set_index('caseId')
df = pd.concat([
	pd.read_json(join('..', 'instruments.json')),
	pd.read_json(join('..', 'anatomies.json')),
])
df['object'] = df['instrument'].fillna(df['anatomy'])

# calculation
df['length'] = df['end'] - df['start'] + 1
df['duration'] = df['length'] / manifest['fps']
caseIds = sorted(df['caseId'].unique())
objects = [Helper.id2anatomy(i) for i in range(Helper.anatomycount())] + [Helper.id2instrument(i) for i in range(Helper.instrumentcount())]
functions = dict()
for i, caseId in enumerate(tqdm(caseIds)):
	for j, obj in enumerate(objects):
		ranges = [(r['start'], r['end']) for _, r in df[(df['caseId'] == caseId) & (df['object'] == obj)].iterrows()]
		length = cs.loc[caseId]['frames']
		function = list()
		if len(ranges) > 0:
			for s, e in ranges:
				function.append((s, e, next(len(GROUPS)-k for k, group in enumerate(GROUPS) if obj in group)))
		#print(i+1, caseId, obj, ranges, function)
		functions[(caseId, obj)] = function


### MERGE IT ###

def check(func1, func2):
	for s1, e1, _ in func1:
		for s2, e2, _ in func2:
			if max(s1, s2) <= min(e1, e2):
				return False
	return True

for n in range(1, 8):
	for ys in itertools.combinations_with_replacement(range(n), len(objects)):
		if len(set(ys)) != n: continue
		merged = list(zip(ys, objects))
		groups = [[obj for y2, obj in merged if y2 == y] for y in range(n)]
		b = True
		for group in groups:
			for obj1, obj2 in itertools.combinations(group, 2):
				for caseId in caseIds:
					if check(functions[(caseId, obj1)], functions[(caseId, obj2)]) is False: break
				else: continue
				b = False
		if b == True: print('Working for n=%d:' % n, groups)


### PLOT IT ###

sns.set_theme(
	style='whitegrid',
	rc={
		'patch.linewidth': 0,
		'axes.grid': False,
		'axes.linewidth': 1,
	}
)
fig, axs = plt.subplots(figsize=(4.4*2, 3.2*2), nrows=30//2, ncols=2)
for i, caseId in enumerate(tqdm(caseIds)):
	ax = axs[i//2, i%2]
	for obj in objects:
		function = functions[(caseId, obj)]
		for x1, x2, y in functions[(caseId, obj)]:
			ax.add_patch(patches.Rectangle(xy=(x1, y-0.5), width=x2-x1+1, height=1, facecolor=Helper.any2color(obj)))
	ax.set_xticks([])
	ax.set_xlim([0, cs.loc[caseId]['frames']])
	ax.set_ylim([0.5, len(GROUPS) + 0.5])
	ax.set_yticks([(len(GROUPS)+1)/2], [i+1])
plt.suptitle('Visualization of Instrument and Anatomy Annotations', size=12)
ncol = 4
legend_order = [a for b in [objects[i::ncol] for i in range(ncol)] for a in b]
plt.legend(
	handles=[patches.Patch(facecolor=Helper.any2color(obj), label=obj) for obj in legend_order],
	ncol=ncol,
	labelspacing=0.4,
	loc='lower center', bbox_to_anchor=(-0.1, -2.57),
)
plt.subplots_adjust(top=0.93, left=0.04, right=0.978, bottom=0.12, hspace=0.1, wspace=0.155)
plt.savefig('instrument-anatomy-annotations.pdf')
plt.show()
