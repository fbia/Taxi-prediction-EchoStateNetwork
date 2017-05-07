import esn_class as esn
from datetime import datetime
from utils import *
import numpy as np
import pandas as pd
from esnTrainAndTestUtils import *


def model_selection2(net_param,
                     data_train, Tcum, data_test,
                     norm_param,
                     params_lists, washout_fraction=0):
    """
      do the parameter optimization with a grid search through the hyper param lists,
      here we use the mean Haversine distance
    :param net_param: dictionary with 'Nu', 'Ny', 'use_lsv','Nr','a','rho','Lambda'.'conn'
    :param data_train:
    :param Tcum: collected training targets
    :param data_test:
    :param norm_param: tuple (minlon, maxlon, minlat, maxlat) for the denormalization
    :param params_lists: tuple (NrList, rhoList, aList, LambdaList)
    :param washout_fraction:
    :return: the dataframe with all the parameter results tabulated
    """
    NrList, rhoList, aList, LambdaList, connList = params_lists
    Lambda = LambdaList[0]
    a = aList[0]
    conn = connList[0]
    param_df = pd.DataFrame()
    # to track the progresses
    totcv = len(NrList) * len(rhoList) * len(aList) * len(LambdaList)
    actcv = 0
    for Nr in NrList:  # reservoir size
        for rho in rhoList:  # expected rho/lsv
            net = esn.Esn(Nu=net_param['Nu'], Ny=net_param['Ny'], Nr=Nr, a=a, conn=conn,
                          rho=rho, Lambda=Lambda, lsv=net_param['use_lsv'])
            for a in aList:  # leaking rate
                # a is used only when computing the new state x, No need to build a new esn
                net.a = a
                # Xcoll [x1,...,xn] col-wise
                Xcoll = fitEsn(data_train, net, washout_fraction)

                for Lambda in LambdaList:  # regularization param
                    # Lambda is used only when computing Wout
                    net.Lambda = Lambda
                    print "Wout train...",
                    t1 = datetime.now()
                    net.train_wout(Xcoll, Tcum, net.Lambda)
                    t2 = datetime.now()
                    print "done in sec",
                    print (t2 - t1).total_seconds()

                    # ######### Train TEST #########
                    MHDtr = applyAndMhd(net, data_train, norm_param, washout_fraction)
                    print "train mean Haversine Distance: km " + str(MHDtr)

                    # ######### VALIDATION TEST #########
                    MHDts = applyAndMhd(net, data_test, norm_param, washout_fraction)
                    print "validation mean Haversine Distance: km " + str(MHDts)

                    actcv += 1
                    print "step " + str(actcv) + "/" + str(totcv) + "-->",
                    param_df = param_df.append(pd.DataFrame(
                        data=dict(zip(('MHDts', 'MHDtr', 'Nr', 'rho', 'a', 'Lambda', 'conn'),
                                      ([MHDts], [MHDtr], [Nr], [rho], [a], [Lambda], [conn])))),
                        ignore_index=True)
                    print "best until now: " + str(param_df['MHDts'].min())
                    if MHDts < param_df['MHDts'].min():
                        print "++++++NEW BEST!"
                        print param_df[param_df['MHDts'].idxmin]
                    print "saving model selection results"
                    param_df.to_csv('tmp.csv', index=False)
                    print"^^^^^^^^^^^^^^^^^^^^^"

    return param_df
