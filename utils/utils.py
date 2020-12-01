import gsw
import numpy as np


def compute_distances(lat, lon):
    """
    Given latitude and longitude values, compute the successive cumulative distances in kilometers.

    :param lat: Array of latitudes.
    :param lon: Array of longitudes.
    :return: Array with the cumulative distances in kilometers.
    """
    if lat.ndim > 1:
        lat = lat.flatten()
    if lon.ndim > 1:
        lon = lon.flatten()
    dx = gsw.distance(lon, lat, 0)/1000
    if len(dx) == 1:
        dx = dx[0]
    dists = np.cumsum(dx)

    return np.concatenate(([0], dists))
