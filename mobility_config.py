import weakref
from typing import Iterator


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
        capacity=float('inf'),
        favorability_distance_profile=None,
        transmission_proba=0.0,
    ):
        assert isinstance(name, str), "`name` must be a string."
        self.name = name
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.capacity = capacity
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
        return max(
            [
                favoribility
                for distance_range, favoribility in self.favorability_distance_profile.items()
                if distance_range[0] <= distance < distance_range[1]
            ]
        )

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


WALKING = MobilityMode(
    name="walking",
    max_distance=30,
    min_distance=0,
    favorability_distance_profile={
        (0, 10): MobilityMode.IsFavorable.VERY,
        (10, 15): MobilityMode.IsFavorable.RATHER,
        (15, 20): MobilityMode.IsFavorable.MODERATELY,
        (20, 25): MobilityMode.IsFavorable.RATHER_NOT,
        (25, 30): MobilityMode.IsFavorable.NO,
    },
    transmission_proba=0.01,
)
BUS = MobilityMode(
    name="bus",
    max_distance=200,
    min_distance=10,
    capacity=30,
    favorability_distance_profile={
        (10, 20): MobilityMode.IsFavorable.MODERATELY,
        (20, 40): MobilityMode.IsFavorable.RATHER,
        (50, 70): MobilityMode.IsFavorable.VERY,
        (70, 100): MobilityMode.IsFavorable.RATHER,
        (100, 150): MobilityMode.IsFavorable.RATHER_NOT,
        (150, 200): MobilityMode.IsFavorable.NO,
    },
    transmission_proba=0.05,
)
CAR = MobilityMode(
    name="car",
    max_distance=1000,
    min_distance=30,
    capacity=1,
    favorability_distance_profile={
        (30, 50): MobilityMode.IsFavorable.NO,
        (50, 70): MobilityMode.IsFavorable.RATHER_NOT,
        (70, 100): MobilityMode.IsFavorable.MODERATELY,
        (100, 200): MobilityMode.IsFavorable.RATHER,
        (200, 500): MobilityMode.IsFavorable.VERY,
        (500, 1000): MobilityMode.IsFavorable.MODERATELY,
    },
    transmission_proba=0.0,
)
