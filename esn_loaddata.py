__author__ = 'brundu'
from utils import *
import pandas as pd
import json


def load_data(ntrips_train, ntrips_test, washout_fraction=0):
    """
      data loading and preprocessing
    :param ntrips_train:
    :param ntrips_test:
    :param washout_fraction:
    :return: the tuple data_train, data_test, Tcollected, (minlon, maxlon, minlat, maxlat)
    """

    """
    note: for the charts the (lat, long) coordinates correspond to (y, x) in the cartesian plane
    in the polyline they are memorized as (long, lat) gps points
    """
    print "loading data...",
    data_train = pd.read_csv('datasets/train.csv', nrows=ntrips_train,
                             usecols=["trip_ID", "POLYLINE"],
                             converters={
                                 'POLYLINE': lambda x: np.array(json.loads(x))})  # [[lon lat]...[lon lat]] layout
    print "done"

    data_train['triplen'] = data_train["POLYLINE"].apply(lambda x: x.shape[0])
    data_train = data_train[data_train['triplen'] > 0]

    # create the training trips image
    plot_trips(data_train, ntrips_train, "taxi_train_trips.png")
    printLenStats(data_train)

    # normalize the trajectories
    minlat, maxlat, minlon, maxlon = computeNormParams(data_train)
    data_train.POLYLINE.apply(lambda x: minmaxnorm2d(x, minlon, maxlon, minlat, maxlat))

    # build the training target set
    data_train['target'] = 's'
    for idx, row in data_train.iterrows():
        l = row["triplen"]
        # v is a tuple (targetlon, targetlat)
        v = row["POLYLINE"][-1]
        # replicate it for the length of each trip, discarding transient% of the trip
        v = [(v[0], v[1])] * (l - int(l * washout_fraction))
        data_train.set_value(idx, 'target', v)
    # cumulate the target [t1t1t1,..,tntntntn] and transpose for col-wise collection
    Tcollected = np.array([y for x in data_train['target'] for y in x]).T
    data_train.drop(['target', 'triplen'], axis=1, inplace=True)

    # loading validation test data
    print "loading test data...",
    data_test = pd.read_csv('datasets/train.csv',
                            nrows=ntrips_test, skiprows=range(1, ntrips_train),
                            usecols=["trip_ID", 'POLYLINE'],
                            converters={'POLYLINE': lambda x: np.array(json.loads(x))})  # [[long lat]...[]]
    print "done"

    # data_test = data_test.sample(1000)
    l = data_test["POLYLINE"].apply(lambda x: x.shape[0])
    data_test = data_test[l > 0]

    # create the test trips image
    plot_trips(data_train, ntrips_train, "taxi_test_trips.png")

    # normalize the trips points wrt the training data
    data_test.POLYLINE.apply(lambda x: minmaxnorm2d(x, minlon, maxlon, minlat, maxlat))

    return data_train, data_test, Tcollected, (minlon, maxlon, minlat, maxlat)
