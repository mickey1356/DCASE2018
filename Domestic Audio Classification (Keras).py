
# coding: utf-8

# In[1]:


##get_ipython().run_line_magic('cd', 'H:/SINS dataset')


# In[2]:


seed = 1356

import numpy as np
np.random.seed(seed)
# from tensorflow import set_random_seed
# set_random_seed(seed)

import sklearn
import cv2
import random
import math
import os
import datetime

from itertools import chain
from collections import Counter
from sklearn.metrics import f1_score

os.chdir("H:/SINS dataset")

# constants
img_folder = 'audio'
img_name = ['_pressure.png', '_spec1.png', '_spec2.png', '_spec3.png', '.wav']

im_size = 64
im_size_flat = im_size * im_size
n_labels = 9
n_channels = 1 # grayscale
sd = np.sqrt(2) / np.sqrt(im_size_flat)

req_improve = 500
update_cnt = 50


# In[3]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[4]:


import librosa

afs, labels, _ = zip(*[line.rstrip('\n').split('\t') for line in open('meta.txt').readlines()])


# In[5]:


import pickle

mfccs = pickle.load(open("mfccs.pickle", "rb"))


# In[6]:


def get_folds(index, n_folds = 4):
    afs, labels, _ = zip(*[line.rstrip('\n').split('\t') for line in open('meta.txt').readlines()])
    afs_imlinks = [img_folder + af[5:-4] + img_name[index] for af in afs]
        
    # get smallest no. of classes
    mincnt = Counter(labels).most_common()[-1][1]
    
    afs_label_sep = {}
    for (af, label) in zip(afs_imlinks, labels):
        if label in afs_label_sep:
            afs_label_sep[label].append(af)
        else:
            afs_label_sep[label] = [af]

    afs_label_sep_sampled = {label:random.sample(afs_label_sep[label], mincnt) for label in afs_label_sep.keys()}

    # partition the graph into n_folds partitions for cross-validation
    folds = {fold+1:[] for fold in range(n_folds)}
    samples = int(math.ceil(mincnt / n_folds))

    for label in afs_label_sep_sampled.keys():
        random.shuffle(afs_label_sep_sampled[label])
        for fold in range(n_folds):
            folds[fold+1] += [(af, label) for af in afs_label_sep_sampled[label][fold*samples:(fold+1)*samples]]

    return list(folds.values())

# 0-indexed get training/testing sets
def get_sets(folds, fold = -1):
    _, labels, _ = zip(*[line.rstrip('\n').split('\t') for line in open('meta.txt').readlines()])
    sces = set(labels)
    sce_int_map = {sce:i+1 for i, sce in enumerate(sces)}
    int_sce_map = {sce_int_map[i]:i for i in sce_int_map.keys()}

    ex_fold = folds[fold]
    ot_fold = folds[0:fold] + folds[fold+1:]
    chainfold = list(chain.from_iterable(ot_fold))
    random.shuffle(ex_fold)
    random.shuffle(chainfold)
    trX, trY = zip(*[(af, label) for (af, label) in chainfold])
    tX, tY = zip(*[(af, label) for (af, label) in ex_fold])
    
    trY = np.array([[1 if int_sce_map[c+1] == label else 0 for c in range(len(sces))] for label in trY])
    tY = np.array([[1 if int_sce_map[c+1] == label else 0 for c in range(len(sces))] for label in tY])
    
    return (trX, trY, tX, tY)


# In[7]:


def get_ims(links):
    return np.array([(1 - cv2.imread(link, 0)/255).reshape((im_size, im_size, 1)) for link in links])
def get_wav(links):
    temp = []
    for link in links:
#         y, sr = librosa.load(link, sr = None, mono = True)
#         temp.append(librosa.feature.mfcc(y=y,sr=sr).T)
        temp.append(mfccs[link])
    return np.array(temp)


# In[8]:


def data_loader(files, batch_size, file_type = 0):
    L = len(files)
    while True:
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            lim = min(L, batch_end)
            if file_type == 0:
                X = get_ims([f[0] for f in files[batch_start:lim]])
                Y = np.array([f[1] for f in files[batch_start:lim]])
                yield (X, Y)
            elif file_type == 1:
                # wav files
                X = get_wav([f[0] for f in files[batch_start:lim]])
                Y = np.array([f[1] for f in files[batch_start:lim]])
                yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size


# In[9]:


def make_model1D():
    model = Sequential()

    model.add(Conv1D(filters = 32, kernel_size = 32, input_shape=(313, 20), activation = 'relu'))
    model.add(Conv1D(filters = 32, kernel_size = 32, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters = 32, kernel_size = 16, activation = 'relu'))
    model.add(Conv1D(filters = 32, kernel_size = 16, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[10]:


def make_model2D_S():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape=(im_size, im_size, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = n_labels, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[11]:


def make_model2D():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape=(im_size, im_size, 1), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', dilation_rate = (3, 3)))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = n_labels, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[12]:


def make_modelLSTM():
    model = Sequential()
    
    model.add(LSTM(units = 64, return_sequences = True, stateful = False, input_shape = (313, 20)))
    model.add(LSTM(units = 64, return_sequences = True, stateful = False))
    model.add(LSTM(units = 64, stateful = False))
    
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = n_labels, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[13]:


def make_modelCom():
    model = Sequential()

    model.add(Conv1D(filters = 32, kernel_size = 32, input_shape=(313, 20), activation = 'relu'))
    model.add(Conv1D(filters = 32, kernel_size = 32, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(units = 64, return_sequences = True, stateful = False))
    model.add(LSTM(units = 64, return_sequences = True, stateful = False))
    model.add(LSTM(units = 64, stateful = False))

    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = n_labels, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[14]:


def cnn(index, use_model, n_folds = 4, batch_size = 50, verbose = 1, sel_fold = 0, epochs = 100, save_model = False, val_split = 0.1, file_type = 0, name = None):
    random.seed(seed)
    model = use_model()
    
    trX, trY, tX, tY = get_sets(get_folds(index, n_folds), sel_fold)
    vsplit = int(len(tX) * val_split)
    val_X = tX[:vsplit]
    val_Y = tY[:vsplit]
    testX = tX[vsplit:]
    testY = tY[vsplit:]
    
    batch_per_epoch = int(math.ceil(len(trX) / batch_size))
    batch_per_val = int(math.ceil(len(val_X) / batch_size))
    batch_per_test = int(math.ceil(len(testX) / batch_size))
    
    if save_model:
        filepath = 'models/model_{index}.hdf5'.format(index = name)
        checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 0, save_best_only = True, mode = 'max')
        model.fit_generator(data_loader(list(zip(trX, trY)), batch_size, file_type),
                            steps_per_epoch = batch_per_epoch,
                            epochs = epochs, verbose = verbose,
                            callbacks = [checkpoint],
                            validation_data = data_loader(list(zip(val_X, val_Y)), batch_size, file_type),
                            validation_steps = batch_per_val)
    else:
        model.fit_generator(data_loader(list(zip(trX, trY)), batch_size, file_type),
                            steps_per_epoch = batch_per_epoch,
                            epochs = epochs,
                            verbose = verbose,
                            validation_data = data_loader(list(zip(val_X, val_Y)), batch_size, file_type),
                            validation_steps = batch_per_val)
    
    preds = model.predict_generator(data_loader(list(zip(testX, testY)), batch_size, file_type),
                                    steps = batch_per_test,
                                    verbose = verbose)
    predsT = np.argmax(preds, axis = 1)
    trueT = np.argmax(testY, axis = 1)
    print("Accuracy: {}. F1: {}".format(np.mean(np.equal(trueT, predsT)), f1_score(trueT, predsT, average = 'macro')))


# In[15]:


from keras.models import load_model

def test_model(index, model, n_folds = 4, batch_size = 50, verbose = 1, sel_fold = 0, val_split = 0.1, file_type = 0):
    random.seed(seed)
    
    trX, trY, tX, tY = get_sets(get_folds(index, n_folds), sel_fold)
    vsplit = int(len(tX) * val_split)
    testX = tX[vsplit:]
    testY = tY[vsplit:]
    
    batch_per_test = int(math.ceil(len(testX) / batch_size))
        
    preds = model.predict_generator(data_loader(list(zip(testX, testY)), batch_size, file_type),
                                    steps = batch_per_test,
                                    verbose = verbose)
    predsT = np.argmax(preds, axis = 1)
    trueT = np.argmax(testY, axis = 1)
    print("Accuracy: {}. F1: {}".format(np.mean(np.equal(trueT, predsT)), f1_score(trueT, predsT, average = 'macro')))


# In[30]:


##cnn(0, make_model2D_S, epochs = 5, save_model = True, name = "test_new")


# In[16]:


model = load_model("models/model_test_new.hdf5")
test_model(0, model)


# In[22]:


##model = load_model("models/model_2D_small_0.hdf5")
##test_model(0, model)
##model = load_model("models/model_2D_small_1.hdf5")
##test_model(1, model)
##model = load_model("models/model_2D_small_2.hdf5")
##test_model(2, model)
##model = load_model("models/model_2D_small_3.hdf5")
##test_model(3, model)
##model = load_model("models/model_1D_norm.hdf5")
##test_model(4, model, file_type = 1)
##model = load_model("models/model_LSTM_norm.hdf5")
##test_model(4, model, file_type = 1)


# In[28]:


##model = load_model("models/model_0.hdf5")
##model.summary()

