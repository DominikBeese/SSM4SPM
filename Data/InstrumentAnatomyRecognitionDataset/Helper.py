''' Author: Dominik Beese
>>> Helper functions for the Cataract-1K dataset (instruments and anatomies)
<<<
'''

INSTRUMENTS = {
	'Slit/Incision Knife': 'SIK',
	'Katena Forceps': 'KF',
	'Gauge': 'Gau',
	'Capsulorhexis Cystotome': 'CC',
	'Capsulorhexis Forceps': 'CF',
	'Phacoemulsification Tip': 'PT',
	'Spatula': 'Spa',
	'Irrigation-Aspiration': 'IA',
	'Lens Injector': 'LI',
}
ANATOMIES = {
	'Iris': 'Iri',
	'Pupil': 'Pup',
	'Intraocular Lens': 'IOL',
}
IDS = list(INSTRUMENTS.keys()) + list(ANATOMIES.keys())[-1:] + list(ANATOMIES.keys())[:-1]

COLORS = {
	'Iris': '#ad3535',
	'Pupil': '#f0ee72',
	'Intraocular Lens': '#c775ce',
	'Slit/Incision Knife': '#79d1e3',
	'Katena Forceps': '#e91e60',
	'Gauge': '#ed80b6',
	'Capsulorhexis Cystotome': '#b8e266',
	'Capsulorhexis Forceps': '#0074b7',
	'Phacoemulsification Tip': '#00a05b',
	'Spatula': '#641d00',
	'Irrigation-Aspiration': '#f49100',
	'Lens Injector': '#6f2caa',
}

def any2color(name: str) -> str:
	color = COLORS.get(name, None)
	if color is None: raise Exception('Unknown instrument: %s' % name)
	return color

def any2id(name: str) -> int:
	return IDS.index(name)

def id2any(index: int) -> str:
	return IDS[index]


def instrumentcount() -> int:
	return len(INSTRUMENTS)

def instrument2abbreviation(name: str) -> str:
	abbreviation = INSTRUMENTS.get(name, None)
	if abbreviation is None: raise Exception('Unknown instrument: %s' % name)
	return abbreviation

def abbreviation2instrument(abbreviation: str) -> str:
	return next(n for n, a in INSTRUMENTS.items() if a == abbreviation)

def instrument2id(name: str) -> int:
	return list(INSTRUMENTS.keys()).index(name)

def id2instrument(index: int) -> str:
	return list(INSTRUMENTS.keys())[index]

def instrument2color(name: str) -> str: return any2color(name)


def anatomycount() -> int:
	return len(ANATOMIES)

def anatomy2abbreviation(name: str) -> str:
	abbreviation = ANATOMIES.get(name, None)
	if abbreviation is None: raise Exception('Unknown anatomy: %s' % name)
	return abbreviation

def abbreviation2anatomy(abbreviation: str) -> str:
	return next(n for n, a in ANATOMIES.items() if a == abbreviation)

def anatomy2id(name: str) -> int:
	return list(ANATOMIES.keys()).index(name)

def id2anatomy(index: int) -> str:
	return list(ANATOMIES.keys())[index]

def anatomy2color(name: str) -> str: return any2color(name)


def abbreviation2any(abbreviation: str) -> str:
	name = next((n for n, a in INSTRUMENTS.items() if a == abbreviation), None)
	if name is not None: return name
	return abbreviation2anatomy(abbreviation)

def any2abbreviation(name: str) -> str:
	try: return instrument2abbreviation(name)
	except: pass
	return anatomy2abbreviation(name)
