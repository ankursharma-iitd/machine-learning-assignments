
# coding: utf-8

# In[159]:

import numpy as np
import os, sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import initializers
from sklearn.metrics import accuracy_score

#define some globals
hash = {}  #mappings of string label with int label
batch_size = 64
num_classes = 20
epochs = 10
test_size = 0.25
img_rows = 28 #number of rows
img_cols = 28 #number of columns
neurons_in_hidden = 1000 #number of neurons in the hideen layer
input_shape = (img_rows, img_cols, 1)
num_pixels = img_rows * img_cols #total length of the feature vector
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)


def preprocess(X):
    return np.array(X, dtype = np.float32)

# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(neurons_in_hidden, input_dim = num_pixels, kernel_initializer=initializers.glorot_normal(seed = 1), activation="sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_normal(seed = 1), activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# loading data and the usual preprocessing
train_directory = './train'
test_directory = './test'
train_y = []
train_X = []
test_X = []
flag = 0
label = 0
for filename in os.listdir(train_directory):
    if filename.endswith(".npy"):
        path_to_file = os.path.join(train_directory, filename)
        X = np.load(path_to_file)
        y = os.path.splitext(filename)[0]
        if(flag == 0):
            train_X = X
            flag = 1
        else:
            train_X = np.vstack((train_X, X))
        for i in range(X.shape[0]):
            train_y.append(label)
        hash[label] = y
        label += 1
test_X = np.load(os.path.join(test_directory, 'test.npy'))

y = keras.utils.to_categorical(np.array(train_y), num_classes)
# y = np.array(train_y)
train_X = preprocess(train_X)
test_X = preprocess(test_X)
# train_X = (train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)).astype('float32')
# test_X = (test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)).astype('float32')

X = (train_X * 1.0) / 255.0
test_X = (test_X * 1.0) / 255.0

# tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size = test_size)

# uncomment the lines below to perform 4 - fold cross validation
# estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=1)
# kfold = KFold(n_splits=4, shuffle=True) #will use 4-fold cross-validation
# results = cross_val_score(estimator, X, y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Get the baseline model
model = baseline_model()

# Fit the model
model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Get the train and test predictions
test_pred = model.predict(test_X)
test_pred = np.argmax(test_pred, axis = 1)

# evaluate the model
scores = model.evaluate(X, y)
print("\ntrain %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#output this onto the file
f = open('neural_net_1000.csv', 'w')
f.write("ID,CATEGORY\n")
for i in range(len(test_pred)):
    f.write(str(i) + "," + hash[test_pred[i]] + "\n")
f.close()