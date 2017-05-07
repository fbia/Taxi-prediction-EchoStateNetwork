__author__ = 'brundu'

from matplotlib.pyplot import *
import pandas as pd
import json
from utils import *
from esnModelSelection import *
import calendar
import datetime
from esnTrainAndTestUtils import *
from esn_class2 import *
from esn_loaddata import *

# ###############  TRAINING PARAMETERS  ############
ntrips_train = 2500  # num of trips to be read from the file each time
ntrips_test = 5000  # num of trips to be read from the file each time
washout_fraction = 0.2
save_results = True
savedir = "/home/brundu/Scrivania/charts/"
paramfile = savedir + "hyp_param_df.csv"

# ###############  DATA LOADING AND PREPROCESSING  ##############
data_train, data_valid, Tcollected, norm_param \
    = load_data(ntrips_train, ntrips_test, washout_fraction)

# ###############  TRAINING  ###################
NrList = [80, 120, 250]  # , 500, 100, 50]  # , 1000]
rhoList = [0.4, 0.6, 0.8, 0.9, 1]  # , 0.90, 0.99, 1.25, 1.50]
aList = [0.4, 0.8, 1]
LambdaList = [1e-4, 1e-3, 1e-2, 0.1]  # , 1e-7, 1e-6, 1e-4]
connList = [30]
hyper_lists = (NrList, rhoList, aList, LambdaList, connList)

Nu = Ny = 2
net_param = {'Nu': Nu, 'Ny': Ny, 'use_lsv': True,
             'Nr': 250, 'a': 1, 'rho': 0.4, 'Lambda': 0.01,
             'conn': 30}

# do the parameters grid search
# and update the net params with the winning ones
model_selection = True
if model_selection:
    hyp_param_df = model_selection2(net_param, data_train, Tcollected,
                                    data_valid, norm_param, hyper_lists,
                                    washout_fraction=washout_fraction)
    print "ended with:",
    winning = hyp_param_df.iloc[hyp_param_df["MHDts"].idxmin()]
    print winning

    if save_results:
        print "saving model selection results"
        hyp_param_df.to_csv(paramfile, index=False)
    print "run the best model"
    net_param['Nr'] = winning['Nr']
    net_param['rho'] = winning['rho']
    net_param['Lambda'] = winning['Lambda']
    net_param['a'] = winning['a']

# ###############  WINNER TRAIN  ###########
# load more data for the test
net = trainEsn(net_param, data_train, Tcollected, washout_fraction)

# ###############  OFFICIAL SUBMISSION GENERATION #################
# applyAndSubmit(net, norm_param, washout_fraction)
# exit()

# ###############  Train TEST #########
MHD = applyAndMhd(net, data_train, norm_param, washout_fraction)
print "mean Haversine Distance: km " + str(MHD)

# ###############  VALIDATION TEST #########
MHD = applyAndMhd(net, data_valid, norm_param, washout_fraction)
print "mean Haversine Distance: km " + str(MHD)

# save the bar plot of Wout
if save_results:
    figure()
    bar(range(2 * int(1 + Nu + net.Nr)), net.W_out.T.flatten())
    title('Output weights $\mathbf{W}^{out}$')
    savefig(savedir + 'Wout.png')

    # figure()
    # plot(X[0:20, 0:200].T)
    # title('Some reservoir activations $\mathbf{x}(n)$')
