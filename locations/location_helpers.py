"""
This file contains parameters about types of locations.

It might look like excessive detail at first, but might help when we want to intervene,
e.g. if we want to know if it makes more sense to close gyms over clubs.
"""

from dataclasses import dataclass
from collections import namedtuple
from typing import List
from functools import lru_cache

import numpy as np
from shapely.geometry.polygon import Polygon

from utilities import py_utils as pyu
from utilities.spatial import GeoCoordinates
from units import Quantity, KM, HOUR, SPACE, TIME


# LocationValue summarizes how valuable a location is for a given category.
# For instance, LocationValue(economic=1., ...) means that a location has
# a large economic value.
LocationValue = namedtuple(
    "LocationValue",
    ["core", "educational", "economic", "fitness", "health", "leisure", "mobility"],
)

# This specifies an almost infinite value for the max number of humans
# in a location.
MAX_CAPACITY = 10e10
MAX_AREA = 10e10


@dataclass
class LocationType(object, metaclass=pyu.InstanceRegistry):
    name: str
    capacity_distribution: List[int] = None
    average_distance_between_humans: float = None
    ventilation: float = None
    confinement_area_distribution: List[float] = None
    location_value: LocationValue = None
    # OpenStreetMap amenity
    amenities: List[str] = None
    # Social
    social_contact_factor: float = None
    surface_prob: List[float] = None
    # Meta
    config_key: str = None

    @property
    def capacity(self):
        if self.capacity_distribution is None:
            return MAX_CAPACITY
        elif isinstance(self.capacity_distribution, (tuple, list)):
            return self.capacity_distribution[-1]
        elif isinstance(self.capacity_distribution, int):
            return self.capacity_distribution
        else:
            raise TypeError

    def sample_capacity(self, rng=None):
        if self.capacity_distribution is None:
            return MAX_CAPACITY
        elif isinstance(self.capacity_distribution, (list, tuple)):
            rng = np.random if rng is None else rng
            return rng.uniform(
                self.capacity_distribution[0], self.capacity_distribution[1]
            )
        elif isinstance(self.capacity_distribution, (int, float)):
            return self.capacity_distribution
        else:
            raise TypeError

    @property
    def confinement_area(self):
        if self.confinement_area_distribution is None:
            return MAX_AREA
        elif isinstance(self.confinement_area_distribution, (list, tuple)):
            return self.confinement_area_distribution[-1]
        elif isinstance(self.confinement_area_distribution, (int, float)):
            return self.confinement_area_distribution
        else:
            raise TypeError

    def sample_confinement_area(self, rng=None):
        if self.confinement_area_distribution is None:
            return MAX_AREA
        elif isinstance(self.confinement_area_distribution, (list, tuple)):
            rng = np.random if rng is None else rng
            return rng.uniform(
                self.confinement_area_distribution[0],
                self.confinement_area_distribution[1],
            )
        elif isinstance(self.confinement_area_distribution, (int, float)):
            return self.confinement_area_distribution
        else:
            raise TypeError

    @classmethod
    def from_config(cls, name, config_key, config=None, **kwargs):
        if config is None:
            from config import LOCATION_DISTRIBUTION

            config = LOCATION_DISTRIBUTION
        kwargs_to_subkey_mapping = {
            "capacity_distribution": "rnd_capacity",
            "social_contact_factor": "social_contact_factor",
            "surface_prob": "surface_prob",
            "confinement_area_distribution": "area",
        }
        for kwarg_key, config_subkey in kwargs_to_subkey_mapping.items():
            kwargs[kwarg_key] = config[config_key][config_subkey]
        return cls(name, config_key=config_key, **kwargs)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


@dataclass
class MobilityMode(LocationType, metaclass=pyu.InstanceRegistry):
    min_distance: Quantity = 0 * KM
    max_distance: Quantity = 10e10 * KM
    speed: Quantity = 1.079e9 * KM / HOUR
    fixed_route: bool = False

    @lru_cache(1000)
    def compute_travel_time(self, distance):
        # isinstance doesn't work here, I checked
        if not distance.__class__.__name__ == "Quantity":
            distance = distance * SPACE
        travel_time = (distance / self.speed).to(TIME)
        return travel_time

    def is_compatible_with_distance(self, distance):
        if not isinstance(distance, Quantity):
            distance = distance * SPACE
        return self.min_distance <= distance <= self.max_distance

    def __hash__(self):
        return hash(self.name)


@dataclass
class LocationSpec(object, metaclass=pyu.InstanceRegistry):
    location_type: LocationType
    coordinates: GeoCoordinates = None
    polygon: Polygon = None
    # OpenStreetMap specific
    osm_id: int = None
    closest_graph_node: int = None
    # Location specific
    location_capacity: int = None
    location_size: float = None

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

    @property
    def area(self):
        # TODO Check for units
        return self.location_size

    @property
    def length(self):
        # TODO Check for units
        return self.location_size

    @classmethod
    def sample(cls, location_type: LocationType, rng=None, **kwargs):
        # Sample things that can be sampled but are not specified in kwargs
        rng = np.random if rng is None else rng
        if "location_capacity" not in kwargs:
            kwargs.update(
                {"location_capacity": location_type.sample_capacity(rng=rng)},
            )
        if "location_size" not in kwargs:
            kwargs.update(
                {"location_size": location_type.sample_confinement_area(rng=rng)},
            )
        return cls(location_type, **kwargs)

    @classmethod
    def get_instances_of_type(
        cls,
        location_type: LocationType = None,
        name: str = None,
        config_key: str = None,
    ):
        if location_type is not None:
            return [
                instance
                for instance in pyu.instances_of(cls)
                if instance.location_type == location_type
            ]
        if name is not None:
            return [
                instance
                for instance in pyu.instances_of(cls)
                if instance.location_type.name == name
            ]
        if config_key is not None:
            return [
                instance
                for instance in pyu.instances_of(cls)
                if instance.location_type.config_key == config_key
            ]
        return []


# ------- Definitions -----------
# -------------------------------
# ------- Location Types --------
VOID = LocationType("void")

# TODO Fill in the details
# Basic
HOUSEHOLD = LocationType.from_config("household", "household")
SENIOR_RESIDENCY = LocationType.from_config("senior_residency", "senior_residency")
OFFICE = LocationType.from_config("office", "workplace")
SCHOOL = LocationType.from_config("school", "school")
UNIVERSITY = LocationType.from_config("university", "misc")

# Fitness
PARK = LocationType.from_config("park", "park")
GYM = LocationType.from_config("gym", "misc")

# Leisure
DINER = LocationType.from_config("diner", "misc")
BAR = LocationType.from_config("bar", "misc")
CLUB = LocationType.from_config("club", "misc")
STADIUM = LocationType.from_config("stadium", "misc", capacity_distribution=(200, 500))

# Necessities
GROCER = LocationType.from_config("grocer", "store")
SUPERMARKET = LocationType.from_config(
    "supermarket", "store", capacity_distribution=(40, 100)
)
MALL = LocationType.from_config("mall", "store", capacity_distribution=(200, 400))
HOSPITAL = LocationType.from_config("hospital", "hospital")

# Transit
# TODO Use the subclass MobilityMode from the other branch
WALK = SIDEWALK = MobilityMode("sidewalk")
SUBWAY = MobilityMode("subway")
BUS = MobilityMode("bus")
CAR = MobilityMode("car")

# -------------------------------
# ----------- Misc --------------
DEFAULT_LOCATION_SPEC = LocationSpec(VOID)
DEFAULT_OSMNX_PLACE = "Plateau Mont-Royal"


if __name__ == "__main__":
    print([inst.name for inst in pyu.instances_of(MobilityMode)])
