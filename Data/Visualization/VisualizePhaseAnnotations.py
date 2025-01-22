''' Author: Dominik Beese
>>> Visualize phase annotations for each video
<<<
'''

from os.path import join
import json
from tqdm.auto import tqdm

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches

import sys
sys.path.append('..')
from PhaseRecognitionDataset import Helper


### CONFIGURATION ###

with open(join('..', 'manifest.json'), 'r', encoding='UTF-8') as file:
	manifest = json.load(file)


### COMPUTE IT ###

cs = pd.read_json(join('..', 'cases.json'))
df = pd.read_json(join('..', 'phases.json'))

# add idle at the beginning
beginning = df.groupby('caseId')['start'].min()
beginning = beginning.to_frame()
beginning['end'] = beginning['start'] - 1
beginning['start'] = 0
beginning['phase'] = 'Idle'
beginning = beginning[beginning['start'] <= beginning['end']]

# add idle at the end
end = df.groupby('caseId')['end'].max()
end = end.to_frame()
end['start'] = end['end'] + 1
end = end.join(cs.set_index('caseId'), on='caseId')
end['end'] = end['frames'] - 1
end['phase'] = 'Idle'
end = end[end['start'] <= end['end']]

df = pd.concat([df, beginning.reset_index(), end.reset_index()])

# calculation
df['length'] = df['end'] - df['start'] + 1
df['duration'] = df['length'] / manifest['fps']
df = df.sort_values(['caseId', 'start']).reset_index()
df = df.groupby('caseId')[['phase', 'duration']].agg(list)


### PLOT IT ###

sns.set_theme(
	style='whitegrid',
	rc={
		'patch.linewidth': 0,
		'axes.grid': False,
		'axes.spines.left': False, 'axes.spines.bottom': False, 'axes.spines.right': False, 'axes.spines.top': False,
	}
)
fig, axs = plt.subplots(figsize=(4.4*2, 3.2*2), nrows=56//2, ncols=2)
for i, idx in enumerate(tqdm(df.index)):
	ax = axs[i//2, i%2]
	dft = pd.DataFrame(df.loc[idx].duration, index=df.loc[idx].phase).transpose()
	dft.plot.barh(stacked=True, legend=None, color=[Helper.phase2color(p) for p in dft.columns], width=3.0, ax=ax)
	ax.set_xticks([])
	ax.set_yticks([0], [i+1])
	ax.set_xlim([0, sum(df.loc[idx].duration)])
plt.suptitle('Visualization of Phase Annotations', size=12)
ncol = 4
legend_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 12, 0]
legend_order = [a for b in [legend_order[i::ncol] for i in range(ncol)] for a in b]
plt.legend(
	handles=[patches.Patch(facecolor=Helper.id2color(i), label=Helper.id2phase(i)) for i in legend_order],
	ncol=ncol,
	labelspacing=0.4,
	loc='lower center',	bbox_to_anchor=(-0.1, -6.2),
)
plt.subplots_adjust(top=0.93, left=0.04, right=0.978, bottom=0.15, hspace=0.1, wspace=0.155)
plt.savefig('phase-annotations.pdf')
plt.show()
