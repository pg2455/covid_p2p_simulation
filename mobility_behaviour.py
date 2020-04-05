from typing import List
from collections import defaultdict, namedtuple
from contextlib import suppress

from config import *
from utils import _draw_random_discreet_gaussian, _normalize_scores

import mobility_config as mcfg
from mobility_engine import Trip
import mobility_utils as mutls

# -------------------------------------------------------------------------------------
# This is here to trick pycharm to give me auto-complete
# noinspection PyUnreachableCode
if False:
    from human import BaseHuman as _object
    from mobility_engine import City
else:
    _object = object


class Visits:
    parks = defaultdict(int)
    stores = defaultdict(int)
    miscs = defaultdict(int)

    @property
    def n_parks(self):
        return len(self.parks)

    @property
    def n_stores(self):
        return len(self.stores)

    @property
    def n_miscs(self):
        return len(self.miscs)


# TODO Make a new class to handle `ExcursionType`
ExcursionType = namedtuple("ExcursionType", ["location", "duration", "request"])


# This class is a mixin and should not inherit from anyone. If it does, it's probably
# because someone (Nasim) forgot to get rid of the superclass...
class MobilityBehaviourMixin(_object):
    """This class should contain everything to do with mobility of humans."""

    def __init__(self, **kwargs):
        super(MobilityBehaviourMixin, self).__init__(**kwargs)
        # Location and related book keeping
        self.household = kwargs.get("household")
        self.workplace = kwargs.get("workplace")
        self.location = self.household
        self.travelled_recently = self.rng.rand() > 0.9
        self.visits = Visits()

        # Mobility habits:
        # ----- Modes -----
        # First we get a random "baseline", which we update based on parameters
        # from config file.
        self.mobility_mode_preference = {
            mode: self.rng.random()
            for mode in mcfg.MobilityMode.get_all_mobility_modes()
        }
        if self.rng.random() > P_HAS_CAR:
            # Account for people who don't own cars
            self.mobility_mode_preference.pop(mcfg.CAR)
        # ----- Exploration -----
        self.rho = kwargs.get("rho", 0.3)
        self.gamma = kwargs.get("gamma", 0.21)
        # ----- Shopping -----
        self.avg_shopping_time = _draw_random_discreet_gaussian(
            AVG_SHOP_TIME_MINUTES, SCALE_SHOP_TIME_MINUTES, self.rng
        )
        self.scale_shopping_time = _draw_random_discreet_gaussian(
            AVG_SCALE_SHOP_TIME_MINUTES, SCALE_SCALE_SHOP_TIME_MINUTES, self.rng
        )
        # TODO: multiple possible days and times & limit these activities in a week
        self.shopping_days = self.rng.choice(range(7))
        self.shopping_hours = self.rng.choice(range(7, 20))
        # ----- Exercise (outdoor) -----
        self.avg_exercise_time = _draw_random_discreet_gaussian(
            AVG_EXERCISE_MINUTES, SCALE_EXERCISE_MINUTES, self.rng
        )
        self.scale_exercise_time = _draw_random_discreet_gaussian(
            AVG_SCALE_EXERCISE_MINUTES, SCALE_SCALE_EXERCISE_MINUTES, self.rng
        )
        self.exercise_days = self.rng.choice(range(7))
        self.exercise_hours = self.rng.choice(range(7, 20))
        # ----- Work -----
        self.avg_working_hours = _draw_random_discreet_gaussian(
            AVG_WORKING_MINUTES, SCALE_WORKING_MINUTES, self.rng
        )
        self.scale_working_hours = _draw_random_discreet_gaussian(
            AVG_SCALE_WORKING_MINUTES, SCALE_SCALE_WORKING_MINUTES, self.rng
        )
        self.work_start_hour = self.rng.choice(range(7, 12))
        # ----- Misc -----
        self.avg_misc_time = _draw_random_discreet_gaussian(
            AVG_MISC_MINUTES, SCALE_MISC_MINUTES, self.rng
        )
        self.scale_misc_time = _draw_random_discreet_gaussian(
            AVG_SCALE_MISC_MINUTES, SCALE_SCALE_MISC_MINUTES, self.rng
        )
        # Compute preferences
        self.parks_preferences = None
        self.stores_preferences = None

    @property
    def lat(self):
        return self.location.lat if self.location else self.household.lat

    @property
    def lon(self):
        return self.location.lon if self.location else self.household.lon

    @property
    def obs_lat(self):
        if LOCATION_TECH == "bluetooth":
            return round(self.lat + self.rng.normal(0, 2))
        else:
            return round(self.lat + self.rng.normal(0, 10))

    @property
    def obs_lon(self):
        if LOCATION_TECH == "bluetooth":
            return round(self.lon + self.rng.normal(0, 2))
        else:
            return round(self.lon + self.rng.normal(0, 10))

    def excursion(self, city: "City", type: str):
        sub_excursions = []

        if type == "shopping":
            location = self._select_location(location_type="stores", city=city)
            duration = _draw_random_discreet_gaussian(
                self.avg_shopping_time, self.scale_shopping_time, self.rng
            )
            sub_excursions.append(
                ExcursionType(location=location, duration=duration, request=True)
            )

        elif type == "exercise":
            location = self._select_location(location_type="park", city=city)
            duration = _draw_random_discreet_gaussian(
                self.avg_exercise_time, self.scale_exercise_time, self.rng
            )
            sub_excursions.append(
                ExcursionType(location=location, duration=duration, request=False)
            )

        elif type == "work":
            location = self.workplace
            duration = _draw_random_discreet_gaussian(
                self.avg_working_hours, self.scale_working_hours, self.rng
            )
            sub_excursions.append(
                ExcursionType(location=location, duration=duration, request=False)
            )

        elif type == "leisure":
            S = 0
            p_exp = 1.0
            while True:
                if self.rng.random() > p_exp:  # return home
                    sub_excursions.append(
                        ExcursionType(
                            location=self.household, duration=60, request=False
                        )
                    )
                    break

                location = self._select_location(location_type="miscs", city=city)
                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                duration = _draw_random_discreet_gaussian(
                    self.avg_misc_time, self.scale_misc_time, self.rng
                )
                sub_excursions.append(
                    ExcursionType(location=location, duration=duration, request=True)
                )
        else:
            raise ValueError(f"Unknown excursion type:{type}")
        return self._execute_excursions(city, sub_excursions)

    def _execute_excursions(self, city: "City", excursions: List[ExcursionType]):
        for excursion in excursions:
            # Plan a trip from current location to the excursion location
            # noinspection PyTypeChecker
            trip = Trip(
                env=self.env,
                human=self,
                source=self.location,
                destination=excursion.location,
                city=city,
            )
            # Take the trip
            trip.take()
            # Now the excursion location may or may not require a request, so...
            requester = excursion.location.request if excursion.request else suppress
            # Stay at the excursion location for however long requested
            with requester() as request:
                if request is not None:
                    yield request
                yield self.env.process(
                    self.at(location=excursion.location, duration=excursion.duration)
                )

    def _compute_preferences(self, city: "City"):
        # This was previously in `City`, but this seems like a better place
        self.stores_preferences = [
            (mutls.compute_geo_distance(self.household, s).to("km").magnitude + 1e-1)
            ** -1
            for s in city.stores
        ]
        self.parks_preferences = [
            (mutls.compute_geo_distance(self.household, s).to("km").magnitude + 1e-1)
            ** -1
            for s in city.parks
        ]

    def _select_location(self, location_type: str, city: "City"):
        """
        Preferential exploration treatment to visit places
        rho, gamma are treated in the paper for normal trips
        Here gamma is multiplied by a factor to supress exploration for parks, stores.
        """
        if location_type in ["park", "stores"] and None in [
            self.parks_preferences,
            self.stores_preferences,
        ]:
            # This must be done once / human
            self._compute_preferences(city)

        if location_type == "park":
            S = self.visits.n_parks
            self.adjust_gamma = 1.0
            pool_pref = self.parks_preferences
            locs = city.parks
            visited_locs = self.visits.parks

        elif location_type == "stores":
            S = self.visits.n_stores
            self.adjust_gamma = 1.0
            pool_pref = self.stores_preferences
            locs = city.stores
            visited_locs = self.visits.stores

        elif location_type == "miscs":
            S = self.visits.n_miscs
            self.adjust_gamma = 1.0
            pool_pref = [
                (mutls.compute_geo_distance(self.location, m).to('km').magnitude + 1e-1) ** -1
                for m in city.miscs
                if m != self.location
            ]
            pool_locs = [m for m in city.miscs if m != self.location]
            locs = city.miscs
            visited_locs = self.visits.miscs

        else:
            raise ValueError(f"Unknown location_type:{location_type}")

        if S == 0:
            p_exp = 1.0
        else:
            p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)

        if self.rng.random() < p_exp and S != len(locs):
            # explore
            cands = [i for i in locs if i not in visited_locs]
            cands = [(loc, pool_pref[i]) for i, loc in enumerate(cands)]
        else:
            # exploit
            cands = [(i, count) for i, count in visited_locs.items()]

        cands, scores = zip(*cands)
        loc = self.rng.choice(cands, p=_normalize_scores(scores))
        visited_locs[loc] += 1
        return loc
