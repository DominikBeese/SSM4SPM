''' Author: Dominik Beese
>>> Helper functions for the Cataract-1K dataset (phases)
<<<
'''

PHASES = {
	'Idle': 'Idl',
	'Incision': 'Inc',
	'Viscoelastic': 'Vis',
	'Capsulorhexis': 'Cap',
	'Hydrodissection': 'Hyd',
	'Phacoemulsification': 'Pha',
	'Irrigation-Aspiration': 'IA',
	'Capsule Polishing': 'CP',
	'Lens Implantation': 'LI',
	'Lens Positioning': 'LP',
	'Anterior-Chamber Flushing': 'ACF',
	'Viscoelastic-Suction': 'VS',
	'Tonifying/Antibiotics': 'TA',
}
ORDER = list(PHASES.keys())[:10] + list(PHASES.keys())[11:9:-1] + list(PHASES.keys())[12:]

COLORS = {
	'Incision': '#62becb',
	'Viscoelastic': '#dd7ca6',
	'Capsulorhexis': '#94c471',
	'Hydrodissection': '#d02f60',
	'Phacoemulsification': '#328f55',
	'Irrigation-Aspiration': '#ecb353',
	'Capsule Polishing': '#c89472',
	'Lens Implantation': '#864687',
	'Lens Positioning': '#9f6c9e',
	'Anterior-Chamber Flushing': '#8a8b8a',
	'Viscoelastic-Suction': '#d65437',
	'Tonifying/Antibiotics': '#1169a3',
	'Idle': '#943036',
}

def phasecount() -> int:
	return len(PHASES)

def phase2abbreviation(name: str) -> str:
	abbreviation = PHASES.get(name, None)
	if abbreviation is None: raise Exception('Unknown phase: %s' % name)
	return abbreviation

def abbreviation2phase(abbreviation: str) -> str:
	return next(n for n, a in PHASES.items() if a == abbreviation)

def phase2id(name: str) -> int:
	return list(PHASES.keys()).index(name)

def phase2order(name: str) -> int:
	return ORDER.index(name)

def id2phase(index: int) -> str:
	return list(PHASES.keys())[index]

def phase2color(name: str) -> str:
	color = COLORS.get(name, None)
	if color is None: raise Exception('Unknown phase: %s' % name)
	return color

def id2abbreviation(index: int) -> str: return phase2abbreviation(id2phase(index))
def abbreviation2id(abbreviation: str) -> int: return phase2id(abbreviation2phase(abbreviation))
def abbreviation2order(abbreviation: str) -> int: return phase2order(abbreviation2phase(abbreviation))
def id2color(index: int) -> str: return phase2color(id2phase(index))
def abbreviation2color(abbreviation: str) -> str: return id2color(abbreviation2id(abbreviation))
