
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os, sys
import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping # early stopping
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.regularizers import l2 # L2-regularisation
from keras.layers.advanced_activations import LeakyReLU, PReLU


# In[2]:


hash = {}  #mappings of string label with int label
batch_size = 64
num_classes = 20
test_size = 0.30
img_rows = 28 #number of rows
img_cols = 28 #number of columns
l2_lambda = 0.0001 #lambda for the l2 regularization
hidden_size = 500
num_of_kernels = 32
num_epochs = 1
input_shape = (img_rows, img_cols, 1)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


# In[3]:


def preprocess(X):
    return np.array(X, dtype = np.float32)


# In[4]:


def baseline_model(kernel_size = 3, num_of_kernels = 32, hidden_size = 500):
    model = Sequential()

    #zero padding the input with strides = 1
    model.add(Conv2D(num_of_kernels, (kernel_size, kernel_size), input_shape = input_shape, activation = 'relu', padding = 'same', strides = (1, 1), kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
    BatchNormalization(axis = -1) #normalise all the means

    #max pooling every 4 pixels to reduce the image size by 1/4th
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #flatten the model
    model.add(Flatten())

    #add one fully connected dense layer
    model.add(Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))

    #get the multilcass classification using softmax regression 
    model.add(Dense(units=20, activation='softmax', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[19]:


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
train_X = preprocess(train_X)
test_X = preprocess(test_X)

X = (train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)).astype('float32')
test_X = (test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)).astype('float32')

X = (X * 1.0) / 255.0
test_X = (test_X * 1.0) / 255.0

tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size = 0.3, random_state=0)


# In[20]:


# # create model
# model = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=64, verbose=1, shuffle = True)


# In[22]:


# # define the grid search parameters
# num_of_kernels = [16, 32, 128, 512]
# kernel_size = [2, 3, 5]
# hidden_size = [100, 500, 1000]
# param_grid = dict(kernel_size=kernel_size, num_of_kernels=num_of_kernels, hidden_size=hidden_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv = 2)


# # In[23]:


# grid_result = grid.fit(X, y, shuffle = True)


# # In[18]:


# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# hidden_size = grid_result.best_params_['hidden_size']
# kernel_size = grid_result.best_params_['kernel_size']
# num_of_kernels = grid_result.best_params_['num_of_kernels']


# In[ ]:
hidden_size = 500
num_of_kernels = 32
kernel_size = 3
model = baseline_model(kernel_size, num_of_kernels, hidden_size)
gen = ImageDataGenerator(rotation_range = 8, width_shift_range = 0.08, shear_range = 0.3, height_shift_range = 0.08, zoom_range = 0.08)
gen.fit(tr_X)
test_gen = ImageDataGenerator()
train_generator = gen.flow(tr_X, tr_y, batch_size = batch_size)
test_generator = test_gen.flow(te_X, te_y, batch_size = batch_size)


# In[ ]:


model.fit_generator(train_generator, steps_per_epoch = tr_X.shape[0]//batch_size, epochs =num_epochs, validation_data = test_generator, validation_steps = te_X.shape[0]//batch_size, verbose=1, shuffle = True)


# In[ ]:


# Fit the model , vary the number of epochs, and the batch size
y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis = 1)


# In[ ]:

# evaluate the model
scores = model.evaluate(X, y)
print("\ntrain %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

f = open('cnn_part_D.csv', 'w')
f.write("ID,CATEGORY\n")
for i in range(len(y_pred)):
    f.write(str(i) + "," + hash[y_pred[i]] + "\n")
f.close()

