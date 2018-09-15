import numpy as np
from random import seed
from random import random
import math,csv
# import visualization as viz
import sys

max_iterations = 2000
learning_rate = 26.6
batch_size = 100
num_layers = None
hidden_layer_list = list()
network = list()
deltas = None
m = 0

def random_weight(lout, lin):
    lout = int(lout)
    lin = int(lin)
    eps_init = (np.sqrt(6)) / (np.sqrt(lout) + np.sqrt(lin))
    ans = np.matrix(np.zeros((lout, lin)))
    np.random.seed(1)
    ans = (np.random.randn(lout, lin) * 2 * eps_init) - eps_init
    return ans

#network is a list of layers, and each layer is a list of dictionaries
def initialise_network(n_features, hidden_list, n_outputs):
    global network
    global num_layers
    for i in range(len(hidden_list) + 1):
        if(i == 0):
            if(len(hidden_list) == 0):
                left_dimension = n_outputs
            else:
                left_dimension = hidden_list[0]
            right_dimension = n_features + 1
        elif(i == len(hidden_list)):
            left_dimension = n_outputs
            right_dimension = hidden_list[len(hidden_list) - 1] + 1
        else:
            left_dimension = hidden_list[i]
            right_dimension = hidden_list[i - 1] + 1
        # theta = np.random.rand(left_dimension, right_dimension)
        theta = random_weight(left_dimension, right_dimension)
        network.append(theta)
    num_layers = len(hidden_list) + 2
    return

def activation(val):
    return 1.0 / (1.0 + np.exp(-val))

def forward_propagate(X, batch_size):
    inputs = list()
    A = np.hstack((np.ones(batch_size).reshape(batch_size, 1), X))
    inputs.append(A)
    for theta in network:
        X = activation(np.dot(A, theta.T))
        A = np.hstack((np.ones(batch_size).reshape(batch_size, 1), X))
        inputs.append(A)
    inputs[num_layers - 1] = inputs[num_layers - 1][:, 1:]
    return inputs

def get_delta(i, inputs, Y):
    if(i == num_layers - 2):
        O = inputs[i + 1]
        delta = np.multiply((Y - O), np.multiply(O, (1 - O)))
    else:
        theta_to_use = network[i + 1][:,1:]
        delta_ahead = deltas[i + 1]
        O = inputs[i + 1][:, 1:]
        delta = np.dot(delta_ahead, theta_to_use) * O * (1 - O)
    return delta

def backpropagation(inputs_A, Y):
    global network
    global deltas
    deltas = [None for i in range(num_layers - 1)]
    for i in range(len(network) - 1, -1, -1):
        somedelta = get_delta(i, inputs_A, Y)
        deltas[i] = somedelta
        network[i] = network[i] + (1.0 / batch_size) * (learning_rate * np.dot(somedelta.T, inputs_A[i]))
    return

# def predict(x):
#     y_out = forward_propagate(x, x.shape[0])
#     y_out = y_out[len(y_out) - 1]
#     new_y = []
#     for i in range(y_out.shape[0]):
#         if(y_out[i] >= 0.5):
#             new_y.append(1)
#         else:
#             new_y.append(0)
#     return np.array(new_y)

def print_accuracy(X, Y):
    A = forward_propagate(X, X.shape[0])
    outputs = (A[len(A) - 1] >= 0.5)
    count = 0
    for i in range(outputs.shape[0]):
        if(int(outputs[i][0]) == Y[i]):
            count += 1
    return ((count * 100.0) / Y.shape[0])

def remap(y):
    for i in range(y.shape[0]):
        if(y[i] == 6):
            y[i] = 0
        else:
            y[i] = 1
    return np.array(y).reshape(y.shape[0], 1)

def main(X, y):
    global hidden_layer_list
    n_features = X.shape[1] #number of input neurons
    n_outputs = 1 #number of neurons in the output layer
    hidden_layer_list = []
    print('\nRANDOMLY INITALISING THE NEURAL NETWORK...')
    initialise_network(n_features, hidden_layer_list, n_outputs) #this method will initialiase the neural network
    print('INITIALISATION DONE!')

    print('\nTRAINING THE NEURAL NETWORK...')
    for iters in range(max_iterations):
        idx = np.random.randint(m, size=batch_size)
        X_new = X[idx, :]
        Y_new = y[idx, :]
        # Y_new = np.array(y_with_2).reshape(m, n_outputs)

        print('\nFORWARD PROPAGATION IN ACTION...')
        A_list = forward_propagate(X_new, X_new.shape[0])
        print('FORWARD PROPAGATION DONE!')

        print('\nBACKWARD PROPAGATION IN ACTION...')
        backpropagation(A_list, Y_new)
        print('BACKPROP DONE!')
    print('NEURAL NETWORK TRAINED!\n')

    test_x = np.loadtxt(
        open("./mnist/test.csv", "rb"), delimiter=",", skiprows=0)
    test_y = test_x[:, nums - 1]
    test_x = (test_x[:, :nums - 1] * 1.0) / 256.0
    test_y = remap(test_y)

    print('TRAINGING ACCURACY : ' + str(print_accuracy(X, y)))
    print('TESTING ACCURACY : ' + str(print_accuracy(test_x, test_y)))

    # #visualising the training decision boundary
    # viz.plot_decision_boundary(lambda x: predict(x), X, y)

    # #visualising the training decision boundary
    # viz.plot_decision_boundary(lambda x: predict(x), test_x, test_y)

    return

if __name__ == '__main__':
    X = np.loadtxt(
        open("./mnist/train.csv", "rb"), delimiter=",", skiprows=0)
    m = X.shape[0]
    nums = X.shape[1]
    y = X[:, nums - 1]
    X = X[:, :nums - 1]
    X = (X * 1.0) / 256.0
    y = remap(y)
    main(X,y)