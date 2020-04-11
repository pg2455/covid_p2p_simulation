"""
This file contains parameters about types of locations.

It might look like excessive detail at first, but might help when we want to intervene,
e.g. if we want to know if it makes more sense to close gyms over clubs.
"""

from dataclasses import dataclass
from typing import Mapping
from collections import namedtuple
from numpy import inf


# LocationValue summarizes how valuable a location is for a given category.
# For instance, LocationValue(economic=1., ...) means that a location has
# a large economic value.
LocationValue = namedtuple(
    "LocationValue",
    ["core", "educational", "economic", "fitness", "health", "leisure", "mobility"],
)


@dataclass
class LocationType(object):
    name: str
    capacity: int = inf
    average_distance_between_humans: float = None
    ventilation: float = None
    confinement_area: float = None
    location_value: LocationValue = None

    def __hash__(self):
        return hash(self.name)


VOID = LocationType('void')

# TODO Fill in the details
# Basic
HOUSEHOLD = LocationType("household")
WORKPLACE = LocationType("workplace")
SCHOOL = LocationType("school")
UNIVERSITY = LocationType("university")

# Fitness
PARK = LocationType("park")
GYM = LocationType("gym")

# Leisure
DINER = LocationType("diner")
BAR = LocationType("bar")
CLUB = LocationType("club")
STADIUM = LocationType("stadium")

# Necessities
GROCER = LocationType("grocer")
SUPERMARKET = LocationType("supermarket")
MALL = LocationType("mall")

# Transit
SIDEWALK = LocationType("sidewalk")
SUBWAY_CAR = LocationType("subway")
