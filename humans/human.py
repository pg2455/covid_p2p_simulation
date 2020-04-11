from collections import deque
from addict import Dict
from config import TICK_MINUTE
from typing import TYPE_CHECKING

from locations import location_types

if TYPE_CHECKING:
    from base import Env
    from locations.location import Location


class Profile(object):

    ACTIVITY_LOCATION_TYPE_MAPPING = {
        "wake": [location_types.HOUSEHOLD],
        "sleep": [location_types.HOUSEHOLD],
        "work": [
            location_types.WORKPLACE,
            location_types.SCHOOL,
            location_types.UNIVERSITY,
        ],
        "exercise": [
            location_types.PARK,
            location_types.SIDEWALK,
            location_types.HOUSEHOLD,
        ],
        "shopping": [
            location_types.GROCER,
            location_types.SUPERMARKET,
            location_types.MALL,
        ],
        "leisure": [
            location_types.DINER,
            location_types.BAR,
            location_types.MALL,
            location_types.PARK,
            location_types.CLUB,
            location_types.STADIUM,
        ],
    }

    def __init__(self):
        # Schedules
        self.schedules = Dict()
        self.location_propensities = Dict()

    @classmethod
    def every_day_at(cls, time):
        return {k: time for k in range(7)}

    @classmethod
    def every_working_day_at(cls, time):
        return {k: time for k in range(5)}

    @classmethod
    def weekends_at(cls, time):
        return {k: time for k in range(5, 7)}

    @classmethod
    def sample_profile(cls):
        pass


class ProtoHuman(object):
    def __init__(self, env: "Env", name: str, favorite_locations: dict = None):
        # Privates
        self._favorite_locations = {}
        # Meta
        self.env = env
        self.name = name
        # Infections
        self.infected = False
        self.infected_at = None
        self.disinfected_at = None
        # Locations
        self.location_history = deque(maxlen=2)
        self.location_entry_timestamp_history = deque(maxlen=2)
        self.bind_favorite_locations(**(favorite_locations or {}))
        # Behaviour
        self.behaviour = Profile.sample_profile()

    def bind_favorite_locations(self, **favorite_locations):
        self._favorite_locations = {
            location_type: [locations]
            if not isinstance(locations, (tuple, list, set))
            else locations
            for location_type, locations in favorite_locations.items()
        }
        return self

    @property
    def location(self):
        return self.location_history[-1]

    @property
    def previous_location(self):
        return self.location_history[-2]

    def run(self):
        while True:
            now = self.env.timestamp
            # TODO
        pass

    def at(self, location: "Location", duration, wait=None):
        if wait is not None:
            yield self.env.timeout(wait / TICK_MINUTE)
        location.enter(self)
        yield self.env.timeout(duration / TICK_MINUTE)
        location.exit(self)

    def expose(self, now):
        # TODO Exposed --> Infected transition
        return self.infect(now)

    def infect(self, now=None):
        if self.infected:
            # Nothing to do here
            return self
        now = now or self.env.timestamp
        self.infected_at = now
        self.infected = True
        return self

    def disinfect(self, now):
        if not self.infected:
            # Nothing to do here
            return self
        now = now or self.env.timestamp
        self.disinfected_at = now
        self.infected = False
        return self

    def __hash__(self):
        return hash(self.name)
