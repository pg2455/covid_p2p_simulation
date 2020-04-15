import uuid
from collections import namedtuple
from dataclasses import dataclass
import datetime

import numpy as np
from simpy import Interrupt
from simpy.core import Infinity

from base import Env
from locations import location_helpers as lty
from utilities import py_utils as pyu

LocationIO = namedtuple(
    "LocationIO",
    [
        "human_name",
        "location_name",
        "io_type",
        "timestamp",
        "human_is_infected",
        "num_infected_humans_at_location",
    ],
)


class LocationFullError(Exception):
    pass


@dataclass(unsafe_hash=True)
class LocationState(object):
    contamination_timestamp: datetime.datetime = datetime.datetime.min
    max_day_contamination: int = 0


class Location(object, metaclass=pyu.InstanceRegistry):
    """Locations are now processes."""

    def __init__(
        self,
        env: Env,
        name: str = None,
        location_spec: lty.LocationSpec = lty.DEFAULT_LOCATION_SPEC,
        location_state: LocationState = None,
        rng: np.random.RandomState = None,
        verbose=False,
    ):
        # Meta data
        self.env = env
        self.name = name if name is not None else uuid.uuid4().hex
        self.spec = location_spec
        self.state = location_state if location_state is not None else LocationState()
        self.rng = rng if rng is not None else np.random
        self.verbose = verbose
        self.now = self.env.timestamp
        # Infection book keeping
        self.last_contaminated = None
        # Entry and exit handling
        self.humans = dict()
        self.entry_queue = []
        self.exit_queue = []
        self.process = self.env.process(self.run())
        # Logging
        self.events = []

    def enter(self, human):
        if len(self.humans) > self.spec.capacity:
            raise LocationFullError
        self.entry_queue.append(human)
        self.process.interrupt()

    def exit(self, human):
        self.exit_queue.append(human)
        self.process.interrupt()

    def run(self):
        while True:
            try:
                # The location sleeps until interrupted
                yield self.env.timeout(Infinity)
            except Interrupt:
                # ^ Wakey wakey.
                # Check the time; we do it once because timedelta in
                # self.env.timestamp consumes a good chuck of the run-time.
                self.now = self.env.timestamp
                # Check who wants to enter
                while self.entry_queue:
                    self.register_human_entry(self.entry_queue.pop())
                # Who infects whom
                self.update_infections()
                # ... and who wants to exit
                while self.exit_queue:
                    self.register_human_exit(self.exit_queue.pop())
                # Back to slumber. We set self.now to None as a tripwire.
                self.now = None

    def update_infections(self):
        # FIXME This is a very naive model.
        #  @Prateek: can you help with this?
        # Infect everyone if anyone is infected in the location.
        if self.infected_human_count > 0:
            for human in self.humans:
                # If the human is already infected, this will not update
                # the infection time-stamp.
                human.expose(self.now)

    def register_human_entry(self, human: "ProtoHuman"):
        if self.verbose:
            print(
                f"Human {human.name} ({'S' if not human.infected else 'I'}) "
                f"entered Location {self.name} at time {self.now} contaminated "
                f"with {self.infected_human_count} infected humans."
            )
        # Set location and timestamps of human
        human.location_history.append(self)
        human.location_entry_timestamp_history.append(self.now)
        # Add human
        self.humans[human] = {
            "was_infected_on_arrival": human.infected,
            "arrived_at": self.now,
        }
        # Record the human entering
        self.events.append(
            LocationIO(
                human_name=human.name,
                location_name=self.name,
                io_type="in",
                timestamp=self.now,
                human_is_infected=human.infected,
                num_infected_humans_at_location=self.infected_human_count,
            )
        )

    @property
    def infected_human_count(self):
        return sum([human.infected for human in self.humans])

    def register_human_exit(self, human: "ProtoHuman"):
        # Record the human exiting
        self.events.append(
            LocationIO(
                human_name=human.name,
                location_name=self.name,
                io_type="out",
                timestamp=self.now,
                human_is_infected=human.infected,
                num_infected_humans_at_location=self.infected_human_count,
            )
        )
        del self.humans[human]
        if self.verbose:
            print(
                f"Human {human.name} ({'S' if not human.infected else 'I'}) "
                f"exited Location {self.name} at time {self.now} contaminated "
                f"with {self.infected_human_count} infected humans."
            )

    def distance_to(self, other):
        if isinstance(other, Location):
            return self.spec.coordinates.distance_to(other.spec.coordinates)
        elif isinstance(other, lty.LocationSpec):
            return self.spec.coordinates.distance_to(other.coordinates)
        elif isinstance(other, lty.GeoCoordinates):
            return self.spec.coordinates.distance_to(other)
        else:
            raise TypeError

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Location) and self.name == other.name

    @classmethod
    def get_instances_of_type(
        cls,
        location_type: lty.LocationType = None,
        name: str = None,
        config_key: str = None,
    ):
        if location_type is not None:
            return [
                instance
                for instance in pyu.instances_of(cls)
                if instance.spec.location_type == location_type
            ]
        if name is not None:
            return [
                instance
                for instance in pyu.instances_of(cls)
                if instance.spec.location_type.name == name
            ]
        if config_key is not None:
            return [
                instance
                for instance in pyu.instances_of(cls)
                if instance.spec.location_type.config_key == config_key
            ]
        return []

    # ---------------------------------------------------------------------------------
    # Glue-code to make this class a drop-in replacement with minimal incisions.
    # These methods should ideally not be used above this section.
    # ---------------------------------------------------------------------------------
    add_human = enter
    remove_human = exit

    @property
    def contamination_timestamp(self):
        return self.state.contamination_timestamp

    @contamination_timestamp.setter
    def contamination_timestamp(self, value):
        self.state.contamination_timestamp = value

    @property
    def max_day_contamination(self):
        return self.state.max_day_contamination

    @max_day_contamination.setter
    def max_day_contamination(self, value):
        self.state.max_day_contamination = value

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    @property
    def lat(self):
        return self.spec.coordinates.lat

    @property
    def lon(self):
        return self.spec.coordinates.lon

    @property
    def area(self):
        return self.spec.area

    @property
    def location_type(self):
        # calling str() on the output does the right thing
        return self.spec.location_type

    @property
    def social_contact_factor(self):
        return self.spec.location_type.social_contact_factor


if __name__ == "__main__":
    from humans.human import ProtoHuman
    import datetime

    env = Env(datetime.datetime(2020, 2, 28, 0, 0))

    L = Location(env, "L", verbose=True)

    A = ProtoHuman(env, "A")
    B = ProtoHuman(env, "B")
    C = ProtoHuman(env, "C").infect(L.now)
    D = ProtoHuman(env, "D")
    E = ProtoHuman(env, "E")
    F = ProtoHuman(env, "F")
    G = ProtoHuman(env, "G")

    env.process(A.at(L, duration=10, wait=0))
    env.process(B.at(L, duration=1, wait=2))
    env.process(C.at(L, duration=4, wait=3))
    env.process(D.at(L, duration=4, wait=4))
    env.process(E.at(L, duration=5, wait=9))
    env.process(F.at(L, duration=5, wait=15))
    env.process(G.at(L, duration=3, wait=16))

    env.run(100)
