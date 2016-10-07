'''
Created on Aug 25, 2016
This class is a decision tree implementation taken from Hal Daume.

@author: km_dh
'''
import numpy as np
import sys
import os
from collections import Counter


class DT(object):

    # any pixel values greater than this will be encoded as a '1',
    # otherwise they will be encoded as a '0'
    PIXEL_THRESHOLD = float(os.environ.get('PIXEL_THRESHOLD', '100'))

    def __init__(self):
        '''
        constructor
        '''

    def res(self, mode='name', model={}, test_case=np.zeros(1), X=np.zeros(1), Y=np.zeros(1), cutoff=1):
        '''
        usage is of the two following:
        learn = DT()
        model = learn.res('train', X=, Y=, cutoff=)
        Y = learn.res('predict', model=, X=)
        '''
        mode = mode.lower()

        if(mode == 'name'):
            return 'DT'

        if(mode == 'train'):
            if(len(X) < 2 or len(Y) < 1 or cutoff < 0):
                print("Error: training requires three arguments: X, Y")
                return 0
            sizeX = X.shape
            sizeY = Y.shape
            if(sizeX[0] != sizeY[0]):
                print("Error: there must be the same number of data points in X and Y")
                return 0
            if sizeY[1] != 1:
                print("Error: Y must have only 1 column")
                return 0
            if(cutoff not in range(1000)):
                print("Error: cutoff must be a positive scalar")
                return 0
            res = {}
            res = self.DTconstruct(X, Y, cutoff)
            return res

        if(mode == 'predict'):
            if(len(model) < 1 or len(test_case) < 1):
                print("Error: prediction requires two arguments: the model and X")
                return 0
            if('isLeaf' not in model.keys()):
                print("Error: model does not appear to be a DT model")
                return 0

            # set up output
            rowCol = test_case.shape
            if(len(rowCol) < 2):
                res = self.DTpredict(model, test_case)
            else:
                N = rowCol[0]
                res = np.zeros(N)
                for n in range(N):
                    ans = self.DTpredict(model, test_case[n, :])
                    res[n] = ans
            return res
        print("Error: unknown DT mode: need train or predict")

    used_feature_list = set()

    def DTconstruct(self, X, Y, cutoff):
        # the Data comes in as X which is NxD and Y which is Nx1.
        # cutoff is a scalar value. We should stop splitting when N is <= cutoff
        #
        # features (X) may not be binary... you should *threshold* them at
        # 0.5, so that anything < 0.5 is a "0" and anything >= 0.5 is a "1"
        #
        # we want to return a *tree*. the way we represent this in our model
        # is that the tree is a Python dictionary.
        #
        # to represent a *leaf* that predicts class 3, we can say:
        #    tree = {}
        #    tree['isLeaf'] = 1
        #    tree['label'] = 3
        #
        # to represent a split node, where we split on feature 5 and then
        # if feature 5 has value 0, we go down the left tree and if feature 5
        # has value 1, we go down the right tree.
        #    tree = {}
        #    tree['isLeaf'] = 0
        #    tree['split'] = 5
        #    tree['left'] = ...some other tree...
        #    tree['right'] = ...some other tree...
        node = {}

        total_feature_count = X.shape[1]
        unvisited_feature_count = total_feature_count - len(self.used_feature_list)
        progress_percent = (float(len(self.used_feature_list)) / float(total_feature_count)) * 100.0
        sys.stdout.write("\rTraining: %.2f%% complete" % (progress_percent))
        sys.stdout.flush()

        label = self.find_most_common_label(Y)

        if self.is_unambiguous(Y):
            node['isLeaf'] = 1
            node['label'] = label
            return node

        elif unvisited_feature_count <= 1 or unvisited_feature_count < cutoff:
            node['isLeaf'] = 1
            node['label'] = label
            return node

        else:
            score = {}
            for i in range(total_feature_count):
                if (i in self.used_feature_list):
                    continue
                yes_subset, no_subset, row_label_mapping = self.get_subsets(X, Y, i)
                score[i] = (self.get_score(yes_subset, no_subset, row_label_mapping))

            # getting max number of correct yes and no's
            feat = self.get_max_score(score)
            self.used_feature_list.add(feat)
            yes_subset_final, no_subset_final, row_label_mapping_final = self.get_subsets(X, Y, feat)

            # pull out a new Y value for each subset
            yes_subset_labels = np.array([row_label_mapping_final['yes_subset']]).T
            no_subset_labels = np.array([row_label_mapping_final['no_subset']]).T

            # leaf node in case on of the subsets is empty
            child_node = {}
            child_node['isLeaf'] = 1
            child_node['label'] = label

            # remove the feature that we just split on from X
            left_node = self.DTconstruct(no_subset_final, no_subset_labels, cutoff) if no_subset_final.shape[0] > 0 else child_node
            right_node = self.DTconstruct(yes_subset_final, yes_subset_labels, cutoff) if yes_subset_final.shape[0] > 0 else child_node

            node['isLeaf'] = 0
            node['split'] = feat
            node['left'] = left_node
            node['right'] = right_node

            return node

    def get_subsets(self, X, Y, feature):
        yes = []
        no = []
        label_dict = {'yes_subset': [], 'no_subset': []}
        index = 0
        for i in X:
            # check value at column (feature)
            # thresholding at 0.5 won't work! Values are from 0 to 255
            if i[feature] >= self.PIXEL_THRESHOLD:
                yes.append(i)
                label_dict['yes_subset'].append(Y[index][0])  # Y is a column, so Y[index] is a 1-element array
            else:
                no.append(i)
                label_dict['no_subset'].append(Y[index][0])
            index += 1
        return [np.array(yes), np.array(no), label_dict]

    def find_most_common_label(self, labels):
        # Counter is a dictionary where the keys are elements in the input array
        # and the values are the counts of those elements
        # Counter().most_common(n) returns an array the n most common key:value pairs,
        # so Counter().most_common(n)[0][0] return the first value (the key) of the first
        # result, i.e. the most common element in the input array
        return Counter(labels.flatten()).most_common(1)[0][0]

    def is_unambiguous(self, labels):
        # turn labes into a sorted array
        sorted_labels = np.msort(labels).flatten()
        # if the first element and last element of the sorted array are the same,
        # then every element in the array is the same
        return sorted_labels[0] == sorted_labels[len(sorted_labels) - 1]

    def get_score(self, yes_subset, no_subset, label_dict):
        yes_label_counter = Counter(label_dict['yes_subset'])
        no_label_counter = Counter(label_dict['no_subset'])
        # The algorithm would guess that all 'yes' labels are the majority yes label,
        # and that all 'no' labels are the majority no label
        majority_yes_count = yes_label_counter.most_common(1)[0][1] if len(label_dict['yes_subset']) > 0 else 0
        majority_no_count = no_label_counter.most_common(1)[0][1] if len(label_dict['no_subset']) > 0 else 0
        total_count = len(label_dict['yes_subset']) + len(label_dict['no_subset'])
        return (float(majority_yes_count) + float(majority_no_count)) / float(total_count)

    def get_max_score(self, score_dict):
        return max(score_dict.iterkeys(), key=(lambda key: score_dict[key]))

    def DTpredict(self, node, X):
        # here we get a tree (in the same format as for DTconstruct) and
        # a single 1xD example that we need to predict with
        if node['isLeaf'] == 1:
            return node['label']
        if X[node['split']] >= self.PIXEL_THRESHOLD:
            return self.DTpredict(node['right'], X)
        else:
            return self.DTpredict(node['left'], X)

    def DTdraw(self, model, level=0):
        indent = ' '
        if model is None:
            return
        print indent*4*level + 'isLeaf: ' + str(model['isLeaf'])
        if model['isLeaf'] == 1:
            print indent*4*level + 'Y: ' + str(model['label'])
            return
        print indent*4*level + 'split ' + str(model['split'])
        left_tree = str(self.DTdraw(model['left'], level+1))
        if left_tree != 'None':
            # print model['left']
            print indent*4*level + 'left: ' + left_tree
        right_tree = str(self.DTdraw(model['right'], level+1))
        if right_tree != 'None':
            # print model['right']
            print indent*4*level + 'right: ' + right_tree
