import numpy as np
import os
import pickle
from sklearn import cross_validation, svm, preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import shelve
from bouts import *
from framemetrics import *
import tailmetrics
from injectargs import *
from plotSVM import *
import matplotlib.pyplot as plt
import csv


### User input
input_path = 'G:\DT-data\\2018\May\May 16\\1st_1st recording' #the folder that contains all the shv data
csvfile = file(input_path+'\csv_test.csv', 'wb')