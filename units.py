import pint

ureg = pint.UnitRegistry()

# Time
S = ureg.s
MIN = ureg.min
HOUR = ureg.hour

# Space
M = ureg.m
KM = ureg.km

# Defaults
SPACE = KM
TIME = HOUR