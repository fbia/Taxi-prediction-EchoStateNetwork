__author__ = 'brundu'
import numpy as np
from scipy import linalg
from datetime import datetime


class Esn:
    def __init__(self, Nu=1, Ny=1, Nr=100, a=1, conn=100, rho=0.99, Lambda=0, win=1, lsv=True):
        """
         Constructor of the net, prepare and initialize the esn
        :param Nu: input units
        :param Ny: output units
        :param Nr: reservoir size
        :param a: leaking rake
        :param conn: % of connectivity
        :param rho: or lsv to which scale the reservoir
        :param Lambda: regularization parameter
        :param win: input weights scaling
        :param lsv: True - use the largest sing value for normalizing W
        :return:
        """
        self.Nu = Nu  # input dim
        self.Ny = Ny  # output dim
        self.Nr = Nr  # reservoir dim
        self.a = a  # leaking rate
        self.connectivity = conn
        self.Lambda = Lambda  # regularization coefficient
        self.x = np.zeros((self.Nr, 1))  # init the reservoir-net-state

        np.random.seed(5)  # fixed
        # input connections, plus the bias unit
        self.W_in = (np.random.rand(Nr, 1 + Nu) - 0.5) * win  # win is the input scaling
        # reservoir connections, make them sparse!
        self.W = np.random.rand(Nr, Nr) - 0.5
        # -5 halves the values
        eps = 0.5 * conn / 100
        # leave only the conn % of the connections
        self.W[abs(self.W) > eps] = 0
        self.W_out = np.zeros((Nr, Nr))

        if not lsv:
            print 'Computing spectral radius...',
            t1 = datetime.now()
            rhoW = np.max(np.abs(linalg.eigvals(self.W)))  # esn necessary condition
            self.W *= rho / rhoW

        else:
            print 'Computing largest sing value...',
            t1 = datetime.now()
            # compute the matrix 2-norm (largest sing. value), esn sufficient condition
            lsv = linalg.norm(self.W, 2)
            self.W *= rho / lsv

        t2 = datetime.now()
        print 'done in ',
        print (t2 - t1).total_seconds(),
        print "sec"

    def fit_data(self, data, washoutLen=0):
        """
         the reservoir-net-state is updated for each training sample,
         warm-up with washoutLen rows
        :param data: the sequence: a matrix with a pattern per row
        :param washoutLen: transient to discard
        :return: the collected activation state
        """
        # allocated memory for the design (collected states) matrix
        # X instead of  [1|U|X]
        X = np.zeros((1 + self.Nu + self.Nr, data.shape[0] - washoutLen))
        # XX = np.zeros((1 + Nu + Nr, 1 + Nu + Nr))
        # YX = np.zeros((Ny, 1 + Nu + Nr))
        # run the reservoir with the data and collect X, skip washoutLen elements "wash out"
        self.x = np.zeros(self.Nr)  # start from the 0 state
        for t in range(washoutLen):
            u = data[t]
            u = np.append(1, u)
            self.x = (1 - self.a) * self.x \
                     + self.a * np.tanh(np.dot(self.W_in, u) + np.dot(self.W, self.x))

        for t in range(washoutLen, data.shape[0]):
            u = data[t]
            u = np.append(1, u)
            self.x = (1 - self.a) * self.x \
                     + self.a * np.tanh(np.dot(self.W_in, u) + np.dot(self.W, self.x))
            # the states matrix X stores also the input u, for direct input-output connections
            X[:, t - washoutLen] = np.append(u, self.x)
            # x1 = np.vstack((1, u, x))
            # XX += np.dot(x1, x1.T)  # cumulative update of the design matrix
            # YX += np.dot(u, x1.T)  # cumulative update
        return X

    def train_wout(self, X, Yt, Lambda):
        """
         train the output Wout, given X and the relative Yt
        :param X: collected (col-wise) training states, [x(1),x(2),..]
        :param Yt: collected (col-wise) target outputs, [y1,y2,..]
        :param Lambda: regularization param
        :return:
        """
        # also input and bias units contribute for the output
        # X instead of  [1|U|X]
        X_T = X.T
        YX = np.dot(Yt, X_T)
        self.W_out = np.dot(YX, linalg.inv(np.dot(X, X_T) + Lambda * np.eye(1 + self.Nu + self.Nr)))
        # W_out = dot( Yt, linalg.pinv(X) )
        # W_out = np.dot(YX, linalg.inv(XX + Lambda * np.eye(1 + Nu + Nr)))  # using the precomputed XX & YX

    def predict(self, data_test, x, washoutLen=0, generative_mode=True):
        """
         the test method does not modify the reservoir-net-state
         it uses all the data skipping the first washoutLen rows
        :param data_test: sequence, matrix with a sample per row, as many cols as sample-attributes
        :param x: is the starting state of the reservoir
        :param washoutLen: num of patterns just to warm up the state
        :param generative_mode: True, the output is used as the next input (self-feeding);
                                False, takes the input from the data
        :return: a matrix with one predicted output per row one foreach test pattern
        """
        # preallocate the output matrix
        Y = np.zeros((data_test.shape[0] - washoutLen, self.Ny))

        for t in range(washoutLen):
            u = data_test[t]
            u = np.append(1, u)
            # just "warm up" the net state x
            x = (1 - self.a) * x + self.a * np.tanh(
                np.dot(self.W_in, u) + np.dot(self.W, x))

        length = data_test.shape[0] - 1
        # start testing with the left sequence
        u = data_test[washoutLen]
        for t in range(washoutLen, data_test.shape[0]):
            u = np.append(1, u)
            x = (1 - self.a) * x \
                + self.a * np.tanh(np.dot(self.W_in, u) + np.dot(self.W, x))

            y = np.dot(self.W_out, np.append(u, x))
            Y[t - washoutLen] = y  # insert the output y row-wise
            if generative_mode:
                # generative mode, feed the output to the next input:
                u = y
            else:
                # predictive mode, take the next input from the data:
                if t < length:
                    u = data_test[t + 1]  # the new input is taken from the test set
        return Y
