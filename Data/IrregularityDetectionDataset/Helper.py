''' Author: Dominik Beese
>>> Helper functions for the Cataract-1K dataset (irregularities)
<<<
'''

IRREGULARITIES = {
	None: 'N',
	'Pupil Contraction': 'PC',
	'IOL Rotation': 'LR',
}

COLORS = {
	None: '#8a8b8a',
	'Pupil Contraction': '#f0ee72',
	'IOL Rotation': '#c775ce',
}

def irregularitycount() -> int:
	return len(IRREGULARITIES)

def irregularity2abbreviation(name: str) -> str:
	abbreviation = IRREGULARITIES.get(name, None)
	if abbreviation is None: raise Exception('Unknown phase: %s' % name)
	return abbreviation

def abbreviation2irregularity(abbreviation: str) -> str:
	return next(n for n, a in IRREGULARITIES.items() if a == abbreviation)

def irregularity2id(name: str) -> int:
	return list(IRREGULARITIES.keys()).index(name)

def id2irregularity(index: int) -> str:
	return list(IRREGULARITIES.keys())[index]

def irregularity2color(name: str) -> str:
	color = COLORS.get(name, None)
	if color is None: raise Exception('Unknown irregularity: %s' % name)
	return color

def id2abbreviation(index: int) -> str: return irregularity2abbreviation(id2irregularity(index))
def abbreviation2id(abbreviation: str) -> int: return irregularity2id(abbreviation2irregularity(abbreviation))
def id2color(index: int) -> str: return irregularity2color(id2irregularity(index))
def abbreviation2color(abbreviation: str) -> str: return id2color(abbreviation2id(abbreviation))
