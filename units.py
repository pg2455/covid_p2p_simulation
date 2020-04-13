import pint

ureg = pint.UnitRegistry()
# Define quantity class with the registry (we'll need it for type annotations)
Quantity = pint.quantity.build_quantity_class(ureg)

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