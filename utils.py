__author__ = 'brundu'

from matplotlib.pyplot import *
import numpy as np


def computeNormParams(data):
    """
      computes the parameters for the normalization of the trajectories
    :param data: 
    :return: a tuple (minlat, maxlat, minlon, maxlon)
    """
    minlat = np.inf
    maxlat = -np.inf
    minlon = np.inf
    maxlon = -np.inf
    for idx, row in data.iterrows():
        trip_lon = row.POLYLINE[:, 0]
        trip_lat = np.array(row['POLYLINE'])[:, 1]
        minlat = min(min(trip_lat), minlat)
        maxlat = max(max(trip_lat), maxlat)
        minlon = min(min(trip_lon), minlon)
        maxlon = max(max(trip_lon), maxlon)
    return minlat, maxlat, minlon, maxlon


def printLenStats(data):
    """
      print the statistics from the field triplen
    :param data:
    :return:
    """
    print "statistics of training trips length: mean",
    print data["triplen"].mean(),  # Mean of values
    print "std",
    print data["triplen"].std(),  # Unbiased standard deviation
    print "var",
    print data["triplen"].var(),  # Unbiased variance
    print "max",
    print data["triplen"].max(),
    print "min",
    print data["triplen"].min()


def doSubmission(data, subfile='datasets/submission.csv'):
    """
     Write the csv file with the format 'trip_ID', 'LATITUDE', 'LONGITUDE'
    :param data: dataframe with at least 'trip_ID', 'LATITUDE', 'LONGITUDE'
    :param subfile: csv file name
    :return:
    """
    import csv
    data.to_csv(subfile,
                columns=['TRIP_ID', 'LATITUDE', 'LONGITUDE'], index=None,
                quoting=csv.QUOTE_NONNUMERIC)


def minmaxnorm2d(v, minv0, maxv0, minv1, maxv1):
    v[:, 0] -= minv0
    v[:, 0] /= maxv0 - minv0
    v[:, 1] -= minv1
    v[:, 1] /= maxv1 - minv1
    return v


def minmaxdenorm2d(v, minv0, maxv0, minv1, maxv1):
    """
     given a 2d vector it denormalize its columns wrt the given parameters
    :param v: 2d vector
    :param minv0:
    :param maxv0:
    :param minv1:
    :param maxv1:
    :return:
    """
    v[:, 0] *= maxv0 - minv0
    v[:, 0] += minv0
    v[:, 1] *= maxv1 - minv1
    v[:, 1] += minv1
    return v


def minmaxdenorm(v, minv, maxv):
    """
     given a 1d vector it denormalize its columns wrt the given parameters
    :param v: 1d vector
    :param minv:
    :param maxv:
    :return:
    """
    v *= maxv - minv
    v += minv
    return v


def HaversineDistance(lat_sub, lon_sub, lat_real, lon_real):
    """
    :param lat_sub: nparray
    :param lon_sub: nparray
    :param lat_real: nparray
    :param lon_real: nparray
    :return: the computed distance in km
    """
    REarth = 6371  # this gives the metrics, now it is set to kilometers
    lat = abs(lat_sub - lat_real) * np.pi / 180
    lon = abs(lon_sub - lon_real) * np.pi / 180
    lat_sub = lat_sub * np.pi / 180
    lat_real = lat_real * np.pi / 180
    a = np.sin(lat / 2) * np.sin(lat / 2) \
        + np.cos(lat_sub) * np.cos(lat_real) * np.sin(lon / 2) * np.sin(lon / 2)
    d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d *= REarth
    return d


def meanHaversineDistance(lat_sub, lon_sub, lat_real, lon_real):
    """
     given the arrays, it computes the mean of the Haversine distances
    :param lat_sub: computed/predicted latitudes for each test trip
    :param lon_sub: computed/predicted longitudes for each test trip
    :param lat_real: real latitudes
    :param lon_real: real longitudes
    :return:
    """
    return np.mean(HaversineDistance(lat_sub, lon_sub, lat_real, lon_real))


def computeMse(data_target, Y):
    """
     compute MSE for the first errorLen time steps
    :param data_target: matrix with a pattern per row
    :param Y: my output, matrix with a pattern per row
    :return: the computed mse
    """
    if data_target.shape != Y.shape:
        print "the shapes does not correspond",
        print data_target.shape,
        print Y.shape
        exit(-1)
    return np.sum(np.square(data_target - Y) / Y.shape[0])


def plot_trips(data, n_trips, name):
    """
      It uses the field "POLYLINE" in the data
    :param data: 
    :param n_trips: 
    :param name: destination file name
    :return:
    """
    savedir = "./"
    bins = 1000
    lat_min, lat_max = 41.04961, 41.24961
    lon_min, lon_max = -8.71099, -8.51099
    z = np.zeros((bins, bins))
    latlon = np.array([(lat, lon)
                       for path in data.POLYLINE
                       for lon, lat in path if len(path) > 0])
    figure()
    z += np.histogram2d(*latlon.T, bins=bins,
                        range=[[lat_min, lat_max],
                               [lon_min, lon_max]])[0]
    log_density = np.log(1 + z)
    title(str(n_trips) + ' Taxi trips')
    imshow(log_density[::-1, :],  # flip vertically
           extent=[lat_min, lat_max, lon_min, lon_max])
    savefig(savedir + name)
    print "img file saved in " + savedir + name


def print_data(data, label, marker):
    c = np.random.rand(3, 1)
    longs = []
    lats = []
    for idx, row in data.iterrows():
        longs.append(row['POLYLINE'][:, 0][-1])
        lats.append(row['POLYLINE'][:, 1][-1])
    scatter(longs, lats, c=c, label=label, marker=marker)
