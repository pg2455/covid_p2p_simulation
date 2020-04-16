import pint

ureg = pint.UnitRegistry()
# Define quantity class with the registry (we'll need it for type annotations)
Quantity = pint.quantity.build_quantity_class(ureg)


def is_quantity(x):
    return x.__class__.__name__ == "Quantity"


def as_float(x, unit):
    if is_quantity(x):
        return x.to(unit).magnitude
    else:
        return x


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
