from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, Point
from math import sin, cos, sqrt, atan2, radians

from units import KM


@dataclass(unsafe_hash=True)
class GeoCoordinates(object):
    lat: float
    lon: float

    def to_shapely(self):
        return Point(self.lon, self.lat)

    @classmethod
    def from_shapely(cls, obj):
        if isinstance(obj, Point):
            lat, lon = obj.y, obj.x
        elif isinstance(obj, Polygon):
            lat, lon = obj.centroid.y, obj.centroid.x
        else:
            raise TypeError
        return cls(lat=lat, lon=lon)

    def distance_to(self, other):
        assert isinstance(other, GeoCoordinates)
        R = 6373.0  # km

        lat1 = radians(self.lat)
        lon1 = radians(self.lon)
        lat2 = radians(other.lat)
        lon2 = radians(other.lon)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance * KM


@dataclass(unsafe_hash=True)
class GeoBox(object):
    north_lat: float
    south_lat: float
    east_lon: float
    west_lon: float

    @property
    def north_west(self):
        return GeoCoordinates(lat=self.north_lat, lon=self.west_lon)

    @property
    def south_west(self):
        return GeoCoordinates(lat=self.south_lat, lon=self.west_lon)

    @property
    def north_east(self):
        return GeoCoordinates(lat=self.north_lat, lon=self.east_lon)

    @property
    def south_east(self):
        return GeoCoordinates(lat=self.south_lat, lon=self.east_lon)

    def area(self):
        north_west = self.north_west
        north_east = self.north_east
        south_west = self.south_west
        delta_lat = north_west.distance_to(south_west)
        delta_lon = north_west.distance_to(north_east)
        area = delta_lat * delta_lon
        return area

    def sample(self, num_samples=1, rng=np.random):
        lats = rng.uniform(self.south_lat, self.north_lat, size=(num_samples,))
        lons = rng.uniform(self.west_lon, self.east_lon, size=(num_samples,))
        geo_locations = [
            GeoCoordinates(float(lat), float(lon)) for lat, lon in zip(lats, lons)
        ]
        return geo_locations


TUEBINGEN_GEOBOX = GeoBox(
    north_lat=48.534169, south_lat=48.481084, west_lon=9.024333, east_lon=9.098703
)

DEFAULT_GEOBOX = TUEBINGEN_GEOBOX