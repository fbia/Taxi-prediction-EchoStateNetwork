__author__ = 'brundu'

from matplotlib.pyplot import *
import esn_class as esn
from datetime import datetime
from utils import *
import pandas as pd


# load the data Mackey
train_rows = 2000
test_rows = 500
# els just to init the reservoir state x
trainWashout = 100
testWashout = 0

print "loading data...",
v = 1  # choose a dataset
data = {
    1: np.loadtxt('datasets/MackeyGlass_t17.txt'),  # MSE = 1.00835041454e-06
    2: np.loadtxt('datasets/henonmap_1.txt'),
    3: np.loadtxt('datasets/myfun.txt'),  # best params(6.92e-08, 500, 1.5, 0.3, 0.0001)
    # 4: np.loadtxt('datasets/gps_01.csv')
}[v]
print "done"

f = pd.read_csv("datasets/MackeyGlass_t17.txt", header=None)
min = f.min()
max = f.max()

# if not necessary do not rebuild
data = f.as_matrix(columns=[0])
nvalues = data.shape[0]
print "values: " + str(nvalues)

# normalize the data and the target
minmaxnorm(data, min, max)

data_train = data[:train_rows]
target_train = data[1:train_rows + 1]
data_test = data[train_rows - testWashout:train_rows + test_rows]
target_test = data[train_rows + 1 - testWashout:train_rows + 1 + test_rows]

figure()
plot(f[:800], c='g')
title('A sample of the Mackey-Glass ${\\tau}=17$ data')

Nu = Ny = 1
NrList = [500, 100, 50]  # , 1000]
rhoList = [1.25, 0.8, 0.90, 0.99, 1.50]
aList = [0.3, 0.5, 0.7, 1.0]
betaList = [1e-8, 1e-6, 1e-4, 0.01]
generative = True
win = 1
res = []

param_names = ('mse', 'Nr', 'rho', 'a', 'Lambda', 'conn')
best_paramd = dict(zip(param_names, (float("inf"), 500, 1.25, 0.3, 1e-8, 30)))

print_all = True
use_lsv = False
model_selection = True
if model_selection:
    for Nr in NrList:  # reservoir size
        for rho in rhoList:  # expected rho
            net = esn.Esn(Nu=Nu, Ny=Ny, Nr=Nr, a=0.3, conn=best_paramd['conn'],
                          rho=rho, Lambda=1e-8, win=win, lsv=use_lsv)

            for a in aList:  # leaking rate
                # a is used only to compute the new state x,
                # I do not need to build a new net
                net.a = a
                X = net.fit_data(data_train, trainWashout)
                x_state = net.x  # backup of the state, every test will start from this one
                Yt = target_train  # the target of u(t) is u(t+1)

                for beta in betaList:  # Lambda is only used for Wout
                    net.train_wout(X, Yt[trainWashout:].T, beta)
                    Y = net.predict(data_test=data_test, x=x_state, washoutLen=testWashout)
                    mse = computeMse(target_test[testWashout:], Y)
                    if mse < best_paramd['mse']:
                        best_paramd.update(dict(zip(('mse', 'Nr', 'rho', 'a', 'Lambda'), (mse, Nr, rho, a, beta))))
                        print best_paramd
                    # res += [(mse, Nr, rho, a, Lambda)]  # best_param(mse, nr, rho, a, Lambda)
                    print 'MSE = ' + str(mse)

# best model retraining
print "best run: ",
print "params ",
print best_paramd

# best_param(mse, nr, rho, a, Lambda)
net = esn.Esn(Nu=Nu, Ny=Ny, Nr=best_paramd['Nr'], a=best_paramd['a'], conn=best_paramd['conn'],
              rho=best_paramd['rho'], Lambda=best_paramd['Lambda'], win=win, lsv=use_lsv)

print "data fitting...",
t1 = datetime.now()
X = net.fit_data(data_train, trainWashout)
t2 = datetime.now()
print" done in sec ",
print (t2 - t1).total_seconds()  # millisec

print "Wout train... done in sec ",
t1 = datetime.now()
net.train_wout(X, target_train[trainWashout:].T, net.Lambda)
t2 = datetime.now()
print (t2 - t1).total_seconds()  # millisec

print "prediction... done in sec ",
t1 = datetime.now()
Y = net.predict(data_test=data_test, x=net.x, washoutLen=testWashout)
t2 = datetime.now()
print (t2 - t1).total_seconds()  # millisec

print "computing mse...",
mse = computeMse(target_test[testWashout:], Y)
print 'MSE = ' + str(mse)

# plot some signals, target and predicted
figure()
plot(target_test[testWashout:], 'g', label='Target signal')  # green
plot(Y, 'b--', label='Free-running predicted signal')  # blue
title('Target and generated signals $y(n)$ starting at $n=0$\n $mse=' + str(mse) + '$')
legend(loc='upper right')
xlabel("time n")
ylabel("y(n)")

if print_all:
    figure()
    plot(X[0:20, 0:200].T)
    title('Some reservoir activations $\mathbf{x}(n)$')

    figure()
    bar(range(1 + net.Nu + net.Nr), net.W_out.T)
    title('Output weights $\mathbf{W}^{out}$')

print "close all windows to finish"

show()
