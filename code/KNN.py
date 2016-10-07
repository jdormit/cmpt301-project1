'''
Created on Aug 29, 2016
This class is a K-Nearest-Neighbors implementation.

@author: km_dh
'''
import numpy as np
from numpy.linalg import norm
from collections import Counter
import sys


class KNN(object):
    '''
    classdocs TODO: Fill this in
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def res(self, mode='name', model={}, test_case=np.zeros(1), X=np.zeros(1), Y=np.zeros(1), cutoff=0):
        '''
        usage is of the two following:
        learn = KNN()
        model = learn.res('train', X=, Y=, cutoff=)
        Y = learn.res('predict', model=, X=)
        '''
        mode = mode.lower()

        if(mode == 'name'):
            return 'KNN'

        if(mode == 'train'):
            if(len(X) < 2 or len(Y) < 1 or cutoff < 1):
                print("Error: training requires three arguments: X, Y, and cutoff")
                return 0
            sizeX = X.shape
            sizeY = Y.shape
            if(sizeX[0] != sizeY[0]):
                print("Error: there must be the same number of data points in X and Y")
                return 0
            if(sizeY[1] != 1):
                print("Error: Y must have only 1 column")
                return 0
            if(cutoff not in range(100)):
                print("Error: cutoff must be a positive scalar")
                return 0
            res = {'X': X, 'Y': Y, 'cutoff': cutoff}
            return res

        if(mode == 'predict'):
            if(len(model) < 1 or len(test_case) < 1):
                print("Error: prediction requires two arguments: the model and X")
                return 0
            if('cutoff' not in model.keys() and 'X' not in model.keys() and 'Y' not in model.keys()):
                print("Error: model does not appear to be a KNN model")
                return 0
            sizeModel = model['X'].shape
            sizeX = test_case.shape
            if(len(sizeX) < 2):
                if(sizeModel[1] != sizeX[0]):
                    print("Error: there must be the same number of features in the model and X")                    
                res = self.KNNpredict(model, test_case)
            else:
                if(sizeModel[1] != sizeX[1]):
                    print("Error: there must be the same number of features in the model and X")
                N = sizeX[0]
                res = np.zeros(N)
                for n in range(N):
                    ans = self.KNNpredict(model, test_case[n, :])
                    res[n] = ans
            return res
        print("Error: unknown KNN mode: need train or predict")

    def KNNpredict(self, model, test_case):
        # model contains X which is NxD, Y which is Nx1, cutoff (really K) which is int. See line 47
        # We return a single value which is the predicted class

        X = model['X']
        Y = model['Y']
        K = model['cutoff']

        # {index: distance}
        distances = {}
        index = 0
        for i in X:
            distances[index] = norm(i - test_case)
            index += 1
        print ''

        # sorted_distances is a list of indexes into X and Y sorted by distance
        sorted_distances = sorted(distances.keys(), key=distances.get)

        count = Counter(Y[sorted_distances[i]][0] for i in range(K))

        return count.most_common(1)[0][0]
