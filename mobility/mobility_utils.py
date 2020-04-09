from collections import namedtuple
import numpy as np

from mobility import mobility_config as mcfg
from math import sin, cos, sqrt, atan2, radians


GeoLocation = namedtuple('Location', ['lat', 'lon'])


def compute_geo_distance(loc1, loc2):
    R = 6373.0  # km

    lat1 = radians(loc1.lat)
    lon1 = radians(loc1.lon)
    lat2 = radians(loc2.lat)
    lon2 = radians(loc2.lon)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance * mcfg.UREG.km


def compute_city_area(city_spec):
    north_west = GeoLocation(lat=city_spec.COORD.NORTH.LAT, lon=city_spec.COORD.WEST.LON)
    south_west = GeoLocation(lat=city_spec.COORD.SOUTH.LAT, lon=city_spec.COORD.WEST.LON)
    north_east = GeoLocation(lat=city_spec.COORD.NORTH.LAT, lon=city_spec.COORD.EAST.LON)
    delta_lat = compute_geo_distance(north_west, south_west)
    delta_lon = compute_geo_distance(north_west, north_east)
    area = delta_lat * delta_lon
    return area


def sample_in_city(city_spec, num_samples=1):
    # FIXME this will not work for some cities in far Eastern Russia
    lats = np.random.uniform(city_spec.COORD.SOUTH.LAT, city_spec.COORD.NORTH.LAT, size=(num_samples,))
    lons = np.random.uniform(city_spec.COORD.WEST.LON, city_spec.COORD.EAST.LON, size=(num_samples,))
    geo_locations = [GeoLocation(lat=float(lat), lon=float(lon)) for lat, lon in zip(lats, lons)]
    return geo_locations
