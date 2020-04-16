from collections import deque
from config import TICK_MINUTE
from typing import TYPE_CHECKING

from humans.human_helpers import HumanProfile
import units
from locations.location_helpers import (
    LocationFullError,
    POLL_INTERVAL_BETWEEN_LOCATION_ENTRY_REQUESTS,
)

if TYPE_CHECKING:
    from base import Env
    from locations.location import Location


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
        self.location_history = deque([None, None], maxlen=2)
        self.location_entry_timestamp_history = deque([None, None], maxlen=2)
        self.bind_favorite_locations(**(favorite_locations or {}))
        # Behaviour
        self.profile = HumanProfile.default_profile()
        self.state = None

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
            yield self.env.timeout(units.as_float(wait / TICK_MINUTE, units.MIN))
        while True:
            try:
                location.enter(self)
                break
            except LocationFullError:
                yield self.env.timeout(
                    POLL_INTERVAL_BETWEEN_LOCATION_ENTRY_REQUESTS / TICK_MINUTE
                )
                continue
        yield self.env.timeout(units.as_float(duration / TICK_MINUTE, units.MIN))
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
