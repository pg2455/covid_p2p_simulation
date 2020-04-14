import uuid
from collections import namedtuple

from simpy import Interrupt
from simpy.core import Infinity

from base import Env
from locations import location_helpers as lty

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


class Location(object):
    """Locations are now processes."""

    def __init__(
        self,
        env: Env,
        name: str = None,
        location_spec: lty.LocationSpec = lty.DEFAULT_LOCATION_SPEC,
        verbose=False,
    ):
        # Meta data
        self.env = env
        self.name = name if name is not None else uuid.uuid4().hex
        self.spec = location_spec
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
                # Check the time; we do it once because timedelta in self.env.timestamp
                # consumes a good chuck of the run-time.
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
        # FIXME This is a very naive model, but it'll be enough for now.
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
