import datetime
import numpy as np

from base import Env
from mobility.mobility_engine import City, Location, PublicTransitStation
from mobility import mobility_config as mcfg
from simulator import Human
from monitors import EventMonitor, SEIRMonitor
from utils import _get_random_age
from config import TICK_MINUTE


class PatchedRNG(np.random.RandomState):
    def random(self):
        return self.uniform(0, 1)


if __name__ == "__main__":
    rng = PatchedRNG(42)
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    n_people = 50
    init_percent_sick = 0.2
    simulation_days = 5

    env = Env(start_time)

    stores = [
        Location.random_location(env, capacity=30, location_type="store")
        for _ in range(5)
    ]
    households = [
        Location.random_location(env, capacity=30, location_type="household")
        for _ in range(20)
    ]
    workplaces = [
        Location.random_location(env, capacity=30, location_type="workplace")
        for _ in range(20)
    ]
    parks = [
        Location.random_location(env, capacity=100, location_type="park")
        for _ in range(2)
    ]
    miscs = [
        Location.random_location(env, capacity=30, location_type="misc")
        for _ in range(10)
    ]
    stations = [
        PublicTransitStation.random_station(
            env=env, mobility_mode=mcfg.BUS, capacity=10
        )
        for _ in range(20)
    ]
    city = City(
        env=env, locations=(stores + households + workplaces + parks + miscs + stations)
    )
    monitors = [EventMonitor(f=120), SEIRMonitor(f=60, verbose=True)]

    humans = [
        Human(
            env=env,
            name=i,
            rng=rng,
            age=_get_random_age(rng),
            infection_timestamp=start_time
            if i < n_people * init_percent_sick
            else None,
            household=rng.choice(households),
            workplace=rng.choice(workplaces),
        )
        for i in range(n_people)
    ]

    for human in humans:
        env.process(human.run(city=city))

    for m in monitors:
        env.process(m.run(env, city=city))
    env.run(until=simulation_days * 24 * 60 / TICK_MINUTE)

