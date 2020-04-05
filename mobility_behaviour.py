from collections import defaultdict

from config import *
from utils import _draw_random_discreet_gaussian, _normalize_scores
# -------------------------------------------------------------------------------------
# This is here to trick pycharm to give me auto-complete
# noinspection PyUnreachableCode
if False:
    from human import BaseHuman as _object
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

# TODO Move to MobilityBehaviourMixin
    @property
    def lat(self):
        return self.location.lat if self.location else self.household.lat

    # TODO Move to MobilityBehaviourMixin
    @property
    def lon(self):
        return self.location.lon if self.location else self.household.lon

    # TODO Move to MobilityBehaviourMixin
    @property
    def obs_lat(self):
        if LOCATION_TECH == "bluetooth":
            return round(self.lat + self.rng.normal(0, 2))
        else:
            return round(self.lat + self.rng.normal(0, 10))

    # TODO Move to MobilityBehaviourMixin
    @property
    def obs_lon(self):
        if LOCATION_TECH == "bluetooth":
            return round(self.lon + self.rng.normal(0, 2))
        else:
            return round(self.lon + self.rng.normal(0, 10))

    # TODO Move to MobilityBehaviourMixin
    def excursion(self, city, type):

        if type == "shopping":
            grocery_store = self._select_location(location_type="stores", city=city)
            t = _draw_random_discreet_gaussian(
                self.avg_shopping_time, self.scale_shopping_time, self.rng
            )
            with grocery_store.request() as request:
                yield request
                yield self.env.process(self.at(grocery_store, t))

        elif type == "exercise":
            park = self._select_location(location_type="park", city=city)
            t = _draw_random_discreet_gaussian(
                self.avg_exercise_time, self.scale_exercise_time, self.rng
            )
            yield self.env.process(self.at(park, t))

        elif type == "work":
            t = _draw_random_discreet_gaussian(
                self.avg_working_hours, self.scale_working_hours, self.rng
            )
            yield self.env.process(self.at(self.workplace, t))

        elif type == "leisure":
            S = 0
            p_exp = 1.0
            while True:
                if self.rng.random() > p_exp:  # return home
                    yield self.env.process(self.at(self.household, 60))
                    break

                loc = self._select_location(location_type="miscs", city=city)
                S += 1
                p_exp = self.rho * S ** (-self.gamma * self.adjust_gamma)
                with loc.request() as request:
                    yield request
                    t = _draw_random_discreet_gaussian(
                        self.avg_misc_time, self.scale_misc_time, self.rng
                    )
                    yield self.env.process(self.at(loc, t))
        else:
            raise ValueError(f"Unknown excursion type:{type}")

# TODO Move to MobilityBehaviourMixin
    def _select_location(self, location_type, city):
        """
        Preferential exploration treatment to visit places
        rho, gamma are treated in the paper for normal trips
        Here gamma is multiplied by a factor to supress exploration for parks, stores.
        """
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
                (compute_distance(self.location, m) + 1e-1) ** -1
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

