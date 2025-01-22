''' Author: Dominik Beese
>>> Visualize total instrument and anatomy durations
<<<
'''

from os.path import join
import json
from tqdm.auto import tqdm

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

import sys
sys.path.append('..')
from InstrumentAnatomyRecognitionDataset import Helper


### CONFIGURATION ###

with open(join('..', 'manifest.json'), 'r', encoding='UTF-8') as file:
	manifest = json.load(file)


### COMPUTE IT ###

cs = pd.read_json(join('..', 'cases.json'))
df = pd.concat([
	pd.read_json(join('..', 'instruments.json')),
	pd.read_json(join('..', 'anatomies.json')),
])
df['object'] = df['instrument'].fillna(df['anatomy'])

# calculation
df['length'] = df['end'] - df['start'] + 1
df['duration'] = df['length'] / manifest['fps']
assert df['caseId'].nunique() == 30
df = df.groupby('object')['duration'].sum() / df['caseId'].nunique()


### PLOT IT ###

sns.set_theme(style='whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3,1]}, figsize=(4.4, 3.2))
order=df.sort_values(ascending=False).index
def plot(ax):
	sns.barplot(
		data=df.reset_index(),
		x='duration', y='object',
		order=order,
		ax=ax,
	)
	for i, bar in enumerate(ax.patches): bar.set_facecolor(Helper.any2color(order[i]))
	ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, i: '%ds' % x))
	ax.set(xlabel=None, ylabel=None)
plot(ax1)
plot(ax2)

ax1.set_xlim(0, 135)
ax2.set_xlim(360, 385)
ax2.set_xticks([375])
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

d = 0.02
kwargs = dict(transform=ax1.transAxes, color='#d2d2d2', clip_on=False)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d*3, +d*3), (-d, +d), **kwargs)
ax2.plot((-d*3, +d*3), (1-d, 1+d), **kwargs)


plt.suptitle('Visibility of Instruments and Anatomies', size=12, y=0.95)
plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.07)
plt.savefig('instrument-anatomy-durations.pdf')
plt.show()
