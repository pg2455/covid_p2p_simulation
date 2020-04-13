"""
This file contains parameters about types of locations.

It might look like excessive detail at first, but might help when we want to intervene,
e.g. if we want to know if it makes more sense to close gyms over clubs.
"""

from dataclasses import dataclass
from collections import namedtuple
from typing import List

from numpy import inf
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

from utilities import py_utils as pyu
from utilities.spatial import GeoCoordinates
from units import Quantity, KM, HOUR


# LocationValue summarizes how valuable a location is for a given category.
# For instance, LocationValue(economic=1., ...) means that a location has
# a large economic value.
LocationValue = namedtuple(
    "LocationValue",
    ["core", "educational", "economic", "fitness", "health", "leisure", "mobility"],
)


@dataclass
class LocationType(object, metaclass=pyu.InstanceRegistry):
    name: str
    capacity: int = inf
    average_distance_between_humans: float = None
    ventilation: float = None
    confinement_area: float = None
    location_value: LocationValue = None
    # OpenStreetMap amenity
    amenities: List[str] = None

    def __hash__(self):
        return hash(self.name)


@dataclass
class MobilityMode(LocationType):
    min_distance: Quantity = 0 * KM
    max_distance: Quantity = 10e10 * KM
    speed: Quantity = 1.079e9 * KM / HOUR
    fixed_route: bool = False


@dataclass
class LocationSpec(object):
    location_type: LocationType
    coordinates: GeoCoordinates = None
    polygon: Polygon = None
    # OpenStreetMap specific
    osm_id: int = None
    closest_graph_node: int = None
    # Location specific
    location_capacity: int = None

    @property
    def capacity(self):
        return (
            self.location_capacity
            if self.location_capacity is not None
            else self.location_type.capacity
        )

    @property
    def lat(self):
        return self.coordinates.lat

    @property
    def lon(self):
        return self.coordinates.lon


# ------- Definitions -----------
# -------------------------------
# ------- Location Types --------
VOID = LocationType("void")

# TODO Fill in the details
# Basic
HOUSEHOLD = LocationType("household")
OFFICE = LocationType("workplace",)
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
# TODO Use the subclass MobilityMode from the other branch
SIDEWALK = LocationType("sidewalk")
SUBWAY = LocationType("subway")
BUS = LocationType("bus")
CAR = LocationType("car")

# -------------------------------
# ----------- Misc --------------
DEFAULT_LOCATION_SPEC = LocationSpec(VOID)
DEFAULT_OSMNX_PLACE = "Plateau Mont-Royal"
