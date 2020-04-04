import weakref
import pint
from addict import Dict

from simpy.core import Infinity


# Measurement units
UREG = pint.UnitRegistry()
# Space
M = UREG.meter
KM = UREG.km
SPACE_UNIT = 50 * M
# Time
H = UREG.hour
MIN = UREG.minute
TIME_UNIT = UREG.hour

# Default lat-long range
DEFAULT_CITY = Dict()
DEFAULT_CITY.COORD.NORTH.LAT = 48.534169
DEFAULT_CITY.COORD.SOUTH.LAT = 48.481084
DEFAULT_CITY.COORD.WEST.LON = 9.024333
DEFAULT_CITY.COORD.EAST.LON = 9.098703

# -------------------------------------------------------------------------------------


# Mobility modes
class MobilityMode(object):
    _all_mobility_modes = set()

    # Favorabilities
    class IsFavorable(object):
        VERY = 0
        RATHER = 1
        MODERATELY = 2
        RATHER_NOT = 3
        NO = 4

    def __init__(
        self,
        name,
        max_distance,
        min_distance=0,
        capacity=Infinity,
        speed=50 * KM / H,
        favorability_distance_profile=None,
        transmission_proba=0.0,
    ):
        assert isinstance(name, str), "`name` must be a string."
        self.name = name
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.capacity = capacity
        self.speed = speed
        if favorability_distance_profile is None:
            self.favorability_distance_profile = {
                (0, float("inf")): self.IsFavorable.VERY
            }
        else:
            self.favorability_distance_profile = favorability_distance_profile
        self.transmission_proba = transmission_proba
        self._all_mobility_modes.add(weakref.ref(self))

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)

    def is_compatible_with_distance(self, distance):
        return self.min_distance <= distance <= self.max_distance

    def favorability_given_distance(self, distance):
        if distance < self.min_distance or distance > self.max_distance:
            return MobilityMode.IsFavorable.NO

        return max(
            [
                favoribility
                for distance_range, favoribility in self.favorability_distance_profile.items()
                if distance_range[0] <= distance < distance_range[1]
            ]
        )

    def travel_time(self, distance):
        return distance / self.speed

    @classmethod
    def get_all_mobility_modes(cls):
        dead = set()
        for ref in cls._all_mobility_modes:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._all_mobility_modes -= dead


# Mobility modes
WALKING = MobilityMode(
    name="walking",
    max_distance=3 * KM,
    min_distance=0 * KM,
    speed=5 * KM / H,
    favorability_distance_profile={
        (0 * KM, 1 * KM): MobilityMode.IsFavorable.VERY,
        (1 * KM, 1.5 * KM): MobilityMode.IsFavorable.RATHER,
        (1.5 * KM, 2 * KM): MobilityMode.IsFavorable.MODERATELY,
        (2 * KM, 2.5 * KM): MobilityMode.IsFavorable.RATHER_NOT,
        (2.5 * KM, 3 * KM): MobilityMode.IsFavorable.NO,
    },
    transmission_proba=0.01,
)
# FIXME Refactor this to use standard units!!
BUS = MobilityMode(
    name="bus",
    max_distance=30 * KM,
    min_distance=500 * M,
    capacity=30,
    speed=20 * KM / H,
    favorability_distance_profile={
        (500 * M, 2 * KM): MobilityMode.IsFavorable.MODERATELY,
        (2 * KM, 3 * KM): MobilityMode.IsFavorable.RATHER,
        (3 * KM, 5 * KM): MobilityMode.IsFavorable.VERY,
        (5 * KM, 8 * KM): MobilityMode.IsFavorable.RATHER,
        (8 * KM, 10 * KM): MobilityMode.IsFavorable.RATHER_NOT,
        (10 * KM, 30 * KM): MobilityMode.IsFavorable.NO,
    },
    transmission_proba=0.05,
)
CAR = MobilityMode(
    name="car",
    max_distance=1000 * KM,
    min_distance=0 * KM,
    # Note that `capacity` here means the number of people that can move by car,
    # which is as much as the road can support.
    capacity=1000,
    speed=50 * KM / H,
    favorability_distance_profile={
        (0 * KM, 1 * KM): MobilityMode.IsFavorable.NO,
        (1 * KM, 3 * KM): MobilityMode.IsFavorable.RATHER_NOT,
        (3 * KM, 5 * KM): MobilityMode.IsFavorable.MODERATELY,
        (5 * KM, 10 * KM): MobilityMode.IsFavorable.RATHER,
        (10 * KM, 1000 * KM): MobilityMode.IsFavorable.VERY,
    },
    transmission_proba=0.0001,
)
