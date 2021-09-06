# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py
#python pacman.py --pacman ClassifierAgent

from __future__ import division
from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import math
import numpy as np
from sklearn import tree


def entropy(data):
    #label_count is the function that will count the classes from the dataset and returns in dict
    labelCount = class_counts(data)
    #sum from label_count from each split
    total = sum(labelCount.values())
    uncertainty  =[labelCount[p]/total for p in labelCount.keys() ]
    #entropy for each split
    impurities = [(-t * math.log(t, 2)) for t in uncertainty]


    return sum(impurities)

#current uncertainty == entropy of the current set i.e.data after split as well as initial dataset @ parentnode
#info gain is basically from split child nodes and parent node
#left and right is the split of the node
def info_gain(left, right, current_uncertainty):
    p_T = len(left) / (len(left) + len(right))
    p_F = 1 - p_T

    # information gain
    SA = current_uncertainty - p_T * entropy(left) - p_F * entropy(right)
    return SA



    #weight = float (len(left))/ (len(left) + right(len))
    #gain = current_uncertainty - (weight * entropy(left)) - ((1-weight) * entropy(right))
    #return gain

#partitioning the dataset
#param training dataset/question is a function which splits the data based on 1 or 0
def partition(colID, data):
    true_rows, false_rows = [], []
    for row in data:
        if row[colID] == 1:
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

#find best split and return it's best index no and best gain
def bestSplit(data):
    #except class column
    n_features = len(data[0])-1
    splitOn = {'index': 0, 'gain': 0}
    #entropy of data before splitting at parent node
    current_uncertainty = entropy(data)

    best_gain = 0

    # iterate through each row of the input data, partitioning the data into 'true' and 'false' groups based on the
    # presence of a 1 or 0 in each index position
    for index in range(n_features):
        true, false = partition(index, data)

         #discard /skip that fail to produce a split
        if len(true) == 0 or len(false) == 0:
            continue

        gain = info_gain(true, false, current_uncertainty)

        if gain >= best_gain:
            best_gain = gain
            splitOn['index'] = index
            splitOn['gain'] = gain
    return splitOn

#class/label counts from training dataset

def class_counts(data):
    counts = dict()
    for row in data:
        if row[-1] not in counts:
            counts[row[-1]] = 0
        counts[row[-1]] += 1
    return counts

#data type dict where it takes counts of classes at a leaf node and returns a classification label.
#if no majority then choose randomly
def class_chooser(data):
    #summing values of dict
    total_values = sum(data.values())

    #dict of output labels i.e. keys/return array
    class_labels = np.sort(data.keys())

    # Calculate class proportions/ratio
    #list of random probability
    random_probability = [data[i]/total_values for i in class_labels]

    # if majority classification, return a weighted random pick of the class labels

    if all(p < 0.5  for p in random_probability):
        #
        return np.random.choice(class_labels, p=random_probability)
    else:
        return class_labels[random_probability.index(max(random_probability))]


def build_tree(data, max_depth=None, level=None):
    if max_depth:
        if not level:
            level = 1
        if level == max_depth:
            return Leaf(data)

        #determine the current best split and it's attributes
        #returns dict of index/question and info gain

        curr_best = bestSplit(data)

        #return current best index/question from the split
        #index is a key from the best split function
        idx = curr_best['index']

        if curr_best['gain'] == 0:
            return Leaf(data)

        #otherwise partition the data
        #through index, and training data
        true, false = partition(idx, data)

        level += 1

        #recursively calling to build the true
        true_branch = build_tree(true, max_depth, level)

        false_branch = build_tree(false, max_depth, level)

        return DecisionNode(idx, true_branch, false_branch)

    # behaviour if no maxDepth
    else:
        splits = bestSplit(data)

        idx = splits['index']

        if splits['gain'] == 0:
            return Leaf(data)

        true, false = partition(idx, data)

        true_branch = build_tree(true)

        false_branch= build_tree(false)
        
        return DecisionNode(idx, true_branch, false_branch)


class Leaf:
    def __init__(self, data):
        self.classes = class_counts(data)


#branch of tree which returns from building tree
class DecisionNode:
    """A Decision Node asks a question/index based on 1 and 0.

        This holds a reference to the question, and to the two child nodes.

        basically it's a child with true and false
    """
#index is a question
    def __init__(self, index, true_branch, false_branch):
        self.index = index
        self.true_branch = true_branch
        self.false_branch = false_branch


def classify(example, node):

    if isinstance(node, Leaf):
        classes = node.classes
        return class_chooser(classes)
#idx is a index return from building tree
    idx = node.index

    if example[idx] == 1:
        return classify(example, node.true_branch)
    else:
        return classify(example, node.false_branch)

#bagging
def bagging(data, n_trees, max_depth=None, sampleSize=0.8):
    bag = []

    for n in range(n_trees):
        # create training data samples of size sampleSize from the original data via sampling with replacement
        idxs = range(int(sampleSize * len(data)))
        sampleIdx = [np.random.choice(idxs) for idx in idxs]
        data = np.array(data)
        sample = data[sampleIdx]

        # train a new tree on each random sample in turn and append the tree to the bag container
        dtree = build_tree(sample, max_depth=max_depth)
        bag.append(dtree)

    return bag


def majorityVote(example, bag):
    """
    Based on an ensemble of trees, classify an example data point using each model in the ensemble, then return the
    majority consensus of all models on that classification
    """
    votes = [classify(example, dtree) for dtree in bag]

    # return modal class
    return max(set(votes), key=votes.count)


# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray)):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)

        # train an ensemble of decision trees based on the training data
        self.classifier = bagging(self.data, 11)


            #targetIndex = len(lineAsArray) - 1
            #self.target.append(lineAsArray[targetIndex])




        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        #
        # Any other code you want to run on startup goes here.
        #
        # You may wish to create your classifier here.
        #
        # *********************************************
        
    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):


        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************

        # Get the actions we can try.
        # Return majority classification from ensemble

        moveNumber = majorityVote(features, self.classifier)
        legal = api.legalActions(state)
        return api.makeMove(self.convertNumberToMove(moveNumber), legal)


        # return api.makeMove(Directions.STOP, legal)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.

        #return api.makeMove(Directions.STOP, legal)

