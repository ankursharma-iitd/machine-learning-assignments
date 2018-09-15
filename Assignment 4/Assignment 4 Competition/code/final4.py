
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os, sys
import keras
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping # early stopping
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.regularizers import l2 # L2-regularisation
from keras.layers.advanced_activations import LeakyReLU, PReLU


# In[2]:


hash = {}  #mappings of string label with int label
num_classes = 20 #total number of classes
test_size = 0.20 #training and validation split
img_rows = 28 #number of rows
img_cols = 28 #number of columns
input_shape = (img_rows, img_cols, 1)

batch_size = 64 # in each iteration, we consider 64 training examples at once
num_epochs = 20 # we iterate twelve times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth = 32 # use 32 kernels in both convolutional layers
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # there will be 128 neurons in both hidden layers

num_train = 100000 # there are 60000 training examples in train set
num_test = 100000 # there are 10000 test examples in test set

l2_lambda = 0.0001 #lambda for the l2 regularization
ens_models = 3 # we will train three separate models on the data

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


# In[3]:


def preprocess(X):
    return np.array(X, dtype = np.float32)


# In[4]:


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

tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size = 0.3)


# In[5]:


model = Sequential()

model.add(Conv2D(conv_depth, (kernel_size, kernel_size), input_shape = input_shape, activation = 'relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis = -1))

model.add(Conv2D(conv_depth, (kernel_size, kernel_size), activation = 'relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(conv_depth, (kernel_size, kernel_size), activation = 'relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis = -1))

model.add(Conv2D(conv_depth, (kernel_size, kernel_size), activation = 'relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis = -1))

model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# model.add(Dropout(drop_prob_1))

#flatten the model
model.add(Flatten())

#feed into a fully connected hidden layer with 512 units
model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis = -1))

model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.5))

#get the multilcass classification using softmax regression 
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(l2_lambda), kernel_initializer='glorot_normal'))
    
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[6]:


gen = ImageDataGenerator(rotation_range = 8, shear_range = 0.3, height_shift_range = 0.08, zoom_range = 0.08)
gen.fit(tr_X)
test_gen = ImageDataGenerator()
train_generator = gen.flow(tr_X, tr_y, batch_size = batch_size)
test_generator = test_gen.flow(te_X, te_y, batch_size = batch_size)


# In[ ]:


model.fit_generator(train_generator, steps_per_epoch = tr_X.shape[0]//batch_size, epochs =15, validation_data = test_generator, validation_steps = te_X.shape[0]//batch_size, verbose=1, shuffle = True, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])


# In[17]:


model.save('my_best_model_4.h5')


# In[ ]:


# model = load_model('my_best_model_4.h5')


# In[18]:


# Fit the model , vary the number of epochs, and the batch size
y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis = 1)


# In[19]:


print(len(y_pred))


# In[20]:


f = open('4.csv', 'w')
f.write("ID,CATEGORY\n")
for i in range(len(y_pred)):
    f.write(str(i) + "," + hash[y_pred[i]] + "\n")
f.close()

