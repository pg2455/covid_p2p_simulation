from addict import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import RandomState


class WeeklyEvent(object):
    def __init__(self, start_hour, stop_hour, days=None, location_type_preference=None):
        self.start_time = start_hour
        self.stop_time = stop_hour
        self.location_type_preference = location_type_preference or Dict()
        if isinstance(days, (list, tuple)):
            self.days = list(days)
        elif isinstance(days, int):
            self.days = [days]
        elif isinstance(days, str):
            if days == "every_day":
                self.every_day()
            elif days == "every_weekday":
                self.every_weekday()
            elif days == "every_weekend":
                self.every_weekend()
            else:
                raise ValueError
        else:
            raise TypeError

    def every_day(self):
        self.days = list(range(0, 7))
        return self

    def every_weekday(self):
        self.days = list(range(0, 5))
        return self

    def every_weekend(self):
        self.days = list(range(5, 7))
        return self

    def sample_location_type(self, rng: "RandomState"):
        assert len(self.location_type_preference) > 0
        return rng.choice(
            a=list(self.location_type_preference.keys()),
            p=list(self.location_type_preference.values()),
        )
