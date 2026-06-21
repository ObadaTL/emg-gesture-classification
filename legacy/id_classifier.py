import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import time
import csv

from datetime import datetime

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import stats
import numpy.matlib
from math import sqrt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn import under_sampling


# Results from data collected
from utils import Configuration, feature_extract, segment_mov, sklearn_training, tensorflow_training

config = Configuration()
now = datetime.now()

emg_data_read = pd.read_csv(config.csv_file)

# For reference
# class_label = ['unmarked data','hand at rest','fist','wrist flexion',
# 'wrist extension','radial deviations','ulnar deviations','extended palm']

titles = ['time', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5'
        'channel6', 'channel7', 'channel8', 'class']

emg_data_read.head()
emg_data = emg_data_read.copy()

# Remove Columns
time = emg_data.pop('time')
# emg_data.pop('channel5')
# emg_data.pop('channel6')
# emg_data.pop('channel7')
# emg_data.pop('channel8')
emg_data = np.array(emg_data)

# The matrix of the data collected (segment dimension, channel no)
row = len(emg_data)
col = len(emg_data[0])
# print("TYPE", type(emg_data))
print(f"The size of the emg data is {row} x {col}")
# print(emg_data)


emg_X, emg_y = emg_data[:, :-1], emg_data[:, -1]
no_channels = len(emg_X[0])
print(emg_X.shape, emg_y.shape)
# feature_no = len(X[0])
# print(feature_no)
print(emg_X)

# t1 = now.strftime("%H:%M:%S")
# print("start time", t1)

feature_mov_list = []
no_channels = config.channel_no
segment_dim = config.segment_dim
increment_dim = config.increment_dim

print('Step {}: Generate Feature Vectors'.format('2'))
print('{}{}: Deriving Features'.format('\t','2.1'))
start21_time = datetime.now()
for time_ in range(0, len(emg_X) - segment_dim, increment_dim):
        for channel in range(no_channels):
                channel_data = emg_X[:, channel]
                data_samplewindow = channel_data[time_: time_ + segment_dim]

                # extract features from segment data
                feature_vec = feature_extract(data_samplewindow, no_channels, config.segment_dim)
                mov = segment_mov(time_, segment_dim, emg_y)
                feature_mov_list += feature_vec

        feature_mov_list += [mov]

end21_time = datetime.now()
print('{}{} Complete ({})'.format('\t\t', '2.1', end21_time - start21_time))
print(len(feature_mov_list))
# print("feature_mov_list", feature_mov_list)
# train_lda(feature_all)

filename = './Data/{}-{}-{}.csv'.format(config.channel_no, config.segment_dim, config.increment_dim)
balanced_file = './Data/balanced{}-{}-{}.csv'.format(config.channel_no, config.segment_dim, config.increment_dim)

fv_list = []
header_list = []
count = 0

no_channels = config.channel_no

# no of channels * feature
no_features_row = no_channels * 13

print('{}Step {}: Recording Features'.format('\t','2.2'))
start22_time = datetime.now()
with open(filename, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        # One row for every feature of all channels in a time sample
        for num in range(1, no_features_row + 1):
                header_num = "feature_" + str(num)
                header_list.append(header_num)
        header_list.append("result")
        wr.writerow(header_list)

        for value in feature_mov_list:
                fv_list.append(value)
                count += 1
                if count == (no_features_row + 1):
                        wr.writerow(fv_list)
                        fv_list = []
                        count = 0
end22_time = datetime.now()
print('{}{} Complete ({})'.format('\t\t', '2.2', end22_time - start22_time))

fv_read = pd.read_csv(filename)
drop_index_list = []
balance_0_1point5 = True
count_values = 0
count_0 = 1
num_0 = 100000

if balance_0_1point5 == True:
        for ind in fv_read.index:
                if fv_read['result'][ind] != 0:
                        count_values += 1
        num_0 = count_values / 7 * 1.5

for ind in fv_read.index:
        if fv_read['result'][ind] == 0:
                count_0 += 1
                if count_0 > num_0 + 1:
                        drop_index_list.append(ind)

fv_read = fv_read.drop(fv_read.index[drop_index_list])
fv_read.to_csv(balanced_file, index=False)

# Extracting feature vector and result data


# prepare file
# Results from Feature vector with result file
# Change the type of classifer that will be used
# Uncomment to use the classifier
# classifer_type = "sklearn"


# fv_read = pd.read_csv(filename)

fv_read.head()
fv_data = fv_read.copy()
fv_data = np.array(fv_data)

fv_x, fv_y = fv_data[:, :-1], fv_data[:, -1]
fv_y = fv_y.astype(int)

print(len(fv_y))

counter = Counter(fv_y)
print(counter)

# Over/undersampling
print('{}Step {}: Isolating Training and Test Data'.format('\t','3'))
start3_time = datetime.now()
fv_x_train, fv_x_test, fv_y_train, fv_y_test = train_test_split(fv_x, fv_y, test_size=0.3, random_state=1, shuffle=True)
end3_time = datetime.now()
print('{}{} Complete ({})'.format('\t\t', '3', end3_time - start3_time))
#
#
# #### LDA

print('{}Step {}: LDA Dimensionality Reduction'.format('\t','4'))
sc = StandardScaler()
fv_x_train = sc.fit_transform(fv_x_train)
fv_x_test = sc.transform(fv_x_test)

print(fv_x_train.shape)
start4_time = datetime.now()

lda = LDA(n_components=4)
fv_x_train = lda.fit_transform(fv_x_train, fv_y_train)
fv_x_test = lda.transform(fv_x_test)
print(fv_x_train.shape)

end4_time = datetime.now()
print('{}{} Complete ({})'.format('\t\t', '4', end4_time - start4_time))
print(type(fv_x_train))
print(fv_x_train.shape, fv_y_train.shape)
print(counter)

print('{}Step {}: Training ({})'.format('\t','5', config.classifier))
start5_time = datetime.now()
if config.classifier == "sklearn":
    sklearn_training(fv_x_train, fv_y_train, fv_x_test, fv_y_test)
    print("sklearn training done")
elif config.classifier == "tensorflow":
    tensorflow_training(fv_x_train, fv_y_train, fv_x_test, fv_y_test)
    print("tensorflow training done")

end5_time = datetime.now()
print('{}{} Complete ({})'.format('\t\t', '5', end5_time - start5_time))