import datetime
import numpy as np

from base import Env
from mobility_engine import City, Location, PublicTransitStation
import mobility_config as mcfg
from human import Human
from utils import _get_random_age


class PatchedRNG(np.random.RandomState):
    def random(self):
        return self.uniform(0, 1)


if __name__ == "__main__":
    rng = PatchedRNG(42)
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    n_people = 100
    init_percent_sick = 0.2

    env = Env(start_time)

    stores = [
        Location.random_location(env, capacity=30, location_type="store")
        for _ in range(5)
    ]
    households = [
        Location.random_location(env, capacity=30, location_type="household")
        for _ in range(40)
    ]
    workplaces = [
        Location.random_location(env, capacity=30, location_type="workplace")
        for _ in range(20)
    ]
    miscs = [
        Location.random_location(env, capacity=30, location_type="misc")
        for _ in range(10)
    ]
    stations = [PublicTransitStation.random_station(env=env, mobility_mode=mcfg.BUS, capacity=10) for _ in range(10)]
    city = City(env=env, locations=(stores + households + workplaces + miscs + stations))

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

    pass
