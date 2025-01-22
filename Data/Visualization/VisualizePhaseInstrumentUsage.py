''' Author: Dominik Beese
>>> Calculate tool and phase cooccurrence
<<<
'''

from tqdm.auto import tqdm
from os.path import join
import pandas as pd
import json

import sys
sys.path.append('..')
from PhaseRecognitionDataset import Helper as PhaseHelper
from InstrumentAnatomyRecognitionDataset import Helper as InstrumentAnatomyHelper

import plotly.express as px


### CONFIGURATION ###

PHASE_ORDER = ['Pha', 'LP', 'VS', 'IA', 'Cap', 'LI', 'Inc', 'TA', 'ACF', 'CP', 'Hyd', 'Vis', 'Idl']
INSTRUMENT_ORDER = ['PT', 'Spa', 'IA', 'CF', 'CC', 'LI', 'SIK', 'KF', 'Gau']


### DO IT ###

with open(join('..', 'manifest.json'), 'r', encoding='UTF-8') as file:
	manifest = json.load(file)

df = pd.read_json(join('..', 'cases.json'))
df = df[(df['phases'] == True) & (df['instruments'] == True) & (df['anatomies'] == True)]
cases = df.set_index('caseId')

phases = pd.read_json(join('..', 'phases.json'))
instruments = pd.read_json(join('..', 'instruments.json'))

# extend phases
result, keys = list(), set()
for _, r in tqdm(phases.iterrows()):
	if r['caseId'] not in cases.index: continue
	for frame in range(r['start'], r['end']+1):
		result.append({'caseId': r['caseId'], 'frame': frame, 'phase': r['phase']})
		keys.add((r['caseId'], frame))
assert len(result) == len(keys)
for caseId, case in cases.iterrows():
	for frame in range(case['frames']):
		if (caseId, frame) not in keys:
			result.append({'caseId': caseId, 'frame': frame, 'phase': 'Idle'})
			keys.add((caseId, frame))
assert len(result) == len(keys)
phases = pd.DataFrame(result)
assert len(phases) == sum(case['frames'] for _, case in cases.iterrows())

# extend instruments
result = list()
for _, r in tqdm(instruments.iterrows()):
	if r['caseId'] not in cases.index: continue
	for frame in range(r['start'], r['end']+1):
		result.append({'caseId': r['caseId'], 'frame': frame, 'instrument': r['instrument']})
instruments = pd.DataFrame(result)

# join
phases = phases.set_index(['caseId', 'frame'])
data = instruments.join(phases, on=['caseId', 'frame'], how='outer')
data['instrument'] = data['instrument'].fillna('No Instrument')

# order
PHASE_ORDER = [PhaseHelper.abbreviation2phase(p) for p in PHASE_ORDER]
INSTRUMENT_ORDER = [InstrumentAnatomyHelper.abbreviation2instrument(i) for i in INSTRUMENT_ORDER] + ['No Instrument']

# plot
data['phase-color'] = data['phase'].map(PhaseHelper.phase2color)
fig = px.parallel_categories(
	data_frame=data,
	dimensions=['phase', 'instrument'],
	color='phase-color',
	labels={'phase': 'Phases', 'instrument': 'Instruments'},
	width=800, height=500,
)
fig.update_layout(font_family='Arial', font_color='#262626', font_size=17)
fig.update_traces(labelfont={'size': 13.7})
fig.update_traces(dimensions=[{'categoryorder': 'array'} for _ in range(2)])
fig.update_traces(dimensions=[{'categoryarray': PHASE_ORDER}, {'categoryarray': INSTRUMENT_ORDER}])
fig.update_layout(title={'text': 'Instrument Usage During Surgical Phases', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 15}})
fig.update_layout(margin={'l': 145, 'r': 135, 't': 50, 'b': 10})
fig.write_image('phase-instrument-usage.pdf')
fig.write_html('phase-instrument-usage.html')
fig.show()
