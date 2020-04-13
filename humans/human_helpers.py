from dataclasses import dataclass
from addict import Dict

from locations import location_helpers as lty
from utilities.date_and_time import WeeklyEvent
from utilities import py_utils as pyu


class HumanState(object):
    PROPERTIES_TO_INITIAL_VALUE = {
        "supply": 1.0,
        "fitness": 1.0,
        "stamina": 1.0,
        "sleep": 1.0,
        "mental_health": 1.0,
    }

    def __init__(self, now):
        self.last_updated = now
        self.levels = Dict(self.PROPERTIES_TO_INITIAL_VALUE)


@dataclass
class ActivityType(object, metaclass=pyu.InstanceRegistry):
    name: str

    def __hash__(self):
        return hash(self.name)


SLEEP = ActivityType("sleep")
WORK = ActivityType("work")
EXERCISE = ActivityType("exercise")
SHOPPING = ActivityType("shopping")
LEISURE = ActivityType("leisure")
COMMUTE = ActivityType("commute")


class HumanProfile(object):

    ACTIVITY_TO_LOCATION_TYPE = {
        SLEEP: [lty.HOUSEHOLD],
        WORK: [lty.OFFICE, lty.SCHOOL, lty.UNIVERSITY,],
        EXERCISE: [lty.PARK, lty.SIDEWALK, lty.HOUSEHOLD,],
        SHOPPING: [lty.GROCER, lty.SUPERMARKET, lty.MALL,],
        LEISURE: [
            lty.DINER,
            lty.BAR,
            lty.MALL,
            lty.PARK,
            lty.CLUB,
            lty.STADIUM,
            lty.HOUSEHOLD,
        ],
        COMMUTE: [lty.SUBWAY, lty.SIDEWALK, lty.BUS, lty.CAR],
    }

    def __init__(self):
        # TODO Age, sex, ...
        self.schedules = Dict()
        self.location_type_propensities = Dict()

    def likes_to_be_at(self, location_type, propensity):
        self.location_type_propensities[location_type] = propensity
        return self

    def does_activity_at_time(
        self, activity, start_hour, stop_hour, days, location_type_preference=None
    ):
        assert activity in self.ACTIVITY_TO_LOCATION_TYPE
        self.schedules[activity] = WeeklyEvent(
            start_hour,
            stop_hour,
            days,
            location_type_preference=location_type_preference,
        )
        return self

    def does_activity_at_location_type(self, activity, location_type_preference):
        self.location_type_propensities[activity] = location_type_preference
        return self

    def suggest_activity(self):
        # TODO
        pass

    @classmethod
    def default_profile(cls):
        profile = cls()
        # Main activities
        profile.does_activity_at_time(SLEEP, 22, 8, "every_weekday")
        profile.does_activity_at_time(WORK, 9, 17, "every_weekday")
        profile.does_activity_at_time(LEISURE, 18, 22, "every_weekday")
        # Weekend!
        profile.does_activity_at_time(SLEEP, 23, 10, "every_weekend")
        profile.does_activity_at_time(LEISURE, 10, 23, "every_weekend")
        # Propensities
        profile.does_activity_at_location_type(SLEEP, {lty.HOUSEHOLD: 1.0})
        profile.does_activity_at_location_type(WORK, {lty.OFFICE: 1.0})
        profile.does_activity_at_location_type(
            SHOPPING, {lty.GROCER: 0.5, lty.SUPERMARKET: 0.5}
        )
        profile.does_activity_at_location_type(
            EXERCISE, {lty.PARK: 0.7, lty.SIDEWALK: 0.3}
        )
        profile.does_activity_at_location_type(
            LEISURE, {lty.HOUSEHOLD: 0.5, lty.CLUB: 0.3, lty.BAR: 0.2},
        )
        return profile
