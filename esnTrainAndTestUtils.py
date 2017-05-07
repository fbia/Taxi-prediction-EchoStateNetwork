__author__ = 'brundu'

import esn_class as esn
from datetime import datetime
import pandas as pd
import numpy as np
from utils import *
import json


def trainEsn(net_param, data_train, Tcoll, washout_fraction=0):
    """
      prepare and train the ESN
    :param net_param: dictionary with 'Nu', 'Ny', 'use_lsv','Nr','a','rho','Lambda'.'conn'
    :param data_train:
    :param Tcoll: col-wise collected train target values
    :return: the trained ESN
    """
    net = esn.Esn(Nu=net_param['Nu'], Ny=net_param['Ny'], Nr=net_param['Nr'],
                  a=net_param['a'], conn=net_param['conn'],
                  rho=net_param['rho'], Lambda=net_param['Lambda'], lsv=net_param['use_lsv'])  # [x1,...,xn]

    net.a = net_param['a']
    Xcoll = fitEsn(data_train, net, washout_fraction)

    print "Wout train...",
    t1 = datetime.now()
    net.train_wout(Xcoll, Tcoll, net.Lambda)
    t2 = datetime.now()
    print "done in sec",
    print (t2 - t1).total_seconds()
    return net


def fitEsn(data_train, net, washout_fraction):
    """
      updates the net states with the training data, and collect its activations states.
    :param data_train:
    :param net:
    :param washout_fraction:
    :return: the collected activation states
    """
    print "data fitting",
    t1 = datetime.now()
    Xcoll = np.zeros((1 + net.Nu + net.Nr, 0))
    # ##########  TRAIN  ################
    # iterate over the taxi trips, the POLYLINE field contains the gps trajectory
    # accumulate the activations states
    zerovec = np.zeros(net.Nr)
    count = 500  # print . every 500 trajectory
    for idx, row in data_train.iterrows():
        # reset the state
        net.x = zerovec
        X = net.fit_data(row.POLYLINE,
                         washoutLen=int(row.POLYLINE.shape[0] * washout_fraction))
        Xcoll = np.append(Xcoll, X, axis=1)

        # print a . every 500 trip, just to know where I am
        count -= 1
        if count == 0:
            print ".",
            count = 500
    t2 = datetime.now()
    print" done in sec ",
    print (t2 - t1).total_seconds()
    return Xcoll


def applyEsn(net, data_test, washout_fraction=0):
    """
        add the columns LONGITUDE and LATITUDE into data_valid
        notice: this column maintain the eventual normalization of the data
    :param net: the ESN to use, already trained
    :param data_test:
    :param washout_fraction:
    :return:
    """
    # ############  TEST  ###################
    # iterate over the taxi trips, the POLYLINE field contains the gps trajectory
    # accumulate the predicted output and the relative target
    # add the columns LONGITUDE and LATITUDE in the test data set
    print "prediction...",
    t1 = datetime.now()
    for idx, row in data_test.iterrows():
        net.x = np.zeros(net.Nr)  # set x(0) = 0
        # take only the last prediction, intermediate predictions just to debug/print
        # I can even use a washout of trip.len-1 and just predict the last
        Y = net.predict(row.POLYLINE, x=net.x, generative_mode=False,
                        washoutLen=int(row.POLYLINE.shape[0] * washout_fraction))
        data_test.set_value(idx, 'LONGITUDE', Y[-1][0])
        data_test.set_value(idx, 'LATITUDE', Y[-1][1])
    print "done in sec",
    print (datetime.now() - t1).total_seconds()


def applyAndMhd(net, data, norm_param, washout_fraction=0):
    """
      the target is computed from the data
    :param net: to be applied
    :param data: data for the test
    :param norm_param: normalization tuple (minlon, maxlon, minlat, maxlat)
    :param washout_fraction: fraction of the trip len to be used as washout steps
    :return: MHD
    """
    minlon, maxlon, minlat, maxlat = norm_param

    applyEsn(net, data, washout_fraction)

    # denormalize the predictions
    minmaxdenorm(data["LATITUDE"], minlat, maxlat)
    minmaxdenorm(data["LONGITUDE"], minlon, maxlon)

    # collect the targets and denormalize them
    target = np.zeros((0, net.Ny))
    for idx, row in data.iterrows():
        target = np.vstack((target, row.POLYLINE[-1]))
    minmaxdenorm2d(target, minlon, maxlon, minlat, maxlat)

    return meanHaversineDistance(
        data["LATITUDE"], data["LONGITUDE"],
        target[:, 1], target[:, 0])


def applyAndSubmit(net, norm_param, washout_fraction=0):
    minlon, maxlon, minlat, maxlat = norm_param
    # load the competition test data
    print "loading competition test data...",
    subdata = pd.read_csv('datasets/taxitest.csv',
                          usecols=["TRIP_ID", 'POLYLINE'],
                          converters={'POLYLINE': lambda x: np.array(json.loads(x))})
    print "done"
    subdata.POLYLINE.apply(lambda x: minmaxnorm2d(x, minlon, maxlon, minlat, maxlat))

    applyEsn(net, subdata, washout_fraction)

    # denormalize the predictions
    minmaxdenorm(subdata["LATITUDE"], minlat, maxlat)
    minmaxdenorm(subdata["LONGITUDE"], minlon, maxlon)

    doSubmission(subdata, 'datasets/submission.csv')
    print "Generation done " + 'datasets/submission.csv'
