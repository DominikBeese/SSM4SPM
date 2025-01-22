''' Author: Dominik Beese
>>> Visualize total phase durations
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
assert df['caseId'].nunique() == 56
df = df.groupby('phase')['duration'].sum() / df['caseId'].nunique()


### PLOT IT ###

sns.set_theme(style='whitegrid')
fig = plt.figure(figsize=(4.4, 3.2))
order=df.sort_values(ascending=False).index
ax = sns.barplot(
	data=df.reset_index(),
	x='duration', y='phase',
	order=order
)
for i, bar in enumerate(ax.patches): bar.set_facecolor(Helper.phase2color(order[i]))
plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, i: '%ds' % x))
plt.suptitle('Duration of Phases', size=12, y=0.95)
plt.xlabel(None)
plt.ylabel(None)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('phase-durations.pdf')
plt.show()
