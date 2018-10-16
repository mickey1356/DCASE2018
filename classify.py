import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os

# from sklearn.model_selection import StratifiedKFold
# from sklearn import preprocessing
from sklearn.metrics import f1_score

seed = 1356
# raw_data = pd.read_pickle("domestic_audio_features.pkl")

os.chdir("H:/SINS dataset")

afs, labels, _ = zip(*[line.rstrip('\n').split('\t') for line in open('meta.txt').readlines()])
afs_pressure = ['img'+af[5:-4]+'_pressure.png' for af in afs]
afs_freq = ['img'+af[5:-4]+'_freq.png' for af in afs]

sces = set(labels)
sce_int_map = {sce:i+1 for i, sce in enumerate(sces)}
int_sce_map = {sce_int_map[i]:i for i in sce_int_map.keys()}

# convert labels into one-hot encoding
y = np.array([[1 if int_sce_map[c+1] == label else 0 for c in range(len(sces))] for label in labels])

# split data such that each label is equally represented
# bal_data = raw_data.groupby("label")
# bal_data = bal_data.apply(lambda x: x.sample(bal_data.size().min(), random_state=seed).reset_index(drop=True))
# y = bal_data["label"]
# X = bal_data.drop("label", axis=1)

# split data into stratified folds for cross validation
# training, validation set is in a ratio of 3:1
# skf = StratifiedKFold(n_splits=4, random_state=seed)

# generate folds and pre-process the data
# std_train, minmax_train, y_train, std_test, minmax_test, y_test
# folds = []
# for train, test in skf.split(X, y):
#     X_train, X_test = X.values[train], X.values[test]
#     y_train, y_test = y.values[train], y.values[test]

#     std_scaler = preprocessing.StandardScaler().fit(X_train)
#     X_train_std = std_scaler.transform(X_train)
#     X_test_std = std_scaler.transform(X_test)

#     minmax_scaler = preprocessing.MinMaxScaler().fit(X_train)
#     X_train_minmax = minmax_scaler.transform(X_train)
#     X_test_minmax = minmax_scaler.transform(X_test)
#     folds.append([X_train_std, X_train_minmax, y_train, X_test_std, X_test_minmax, y_test])

# for fold in folds:
#     pass
