import numpy as np
from random import seed
from random import random
import math,csv
import visualization as viz

max_iterations = 2000
# learning_rate = 100 #used where there are 1 units in the hidden layer
# learning_rate = 60 #used where there are 2 units in the hidden layer
# learning_rate = 40  #used where there are 3 units in the hidden layer
# learning_rate = 30  #used where there are 10 units in the hidden layer
# learning_rate = 20 #used where there are 20 units in the hidden layer
learning_rate = 10 #used where there are 40 units in the hidden layer
# learning_rate = 26.6  #used where there are 5 units in the hidden layer
batch_size = 0
num_layers = None
network = list()
deltas = None
m = 0


def random_weight(lout, lin):
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

def predict(x):
    y_out = forward_propagate(x, x.shape[0])
    y_out = y_out[len(y_out) - 1]
    new_y = []
    for i in range(y_out.shape[0]):
        if(y_out[i] >= 0.5):
            new_y.append(1)
        else:
            new_y.append(0)
    return np.array(new_y)

def print_accuracy(X, Y):
    A = forward_propagate(X, X.shape[0])
    outputs = (A[len(A) - 1] >= 0.5)
    count = 0
    for i in range(outputs.shape[0]):
        if(int(outputs[i][0]) == Y[i]):
            count += 1
    return ((count * 100.0) / Y.shape[0])

def main(X, y):
    global batch_size
    hidden_layer_list = [40] #only one hidden layer with 5 neurons
    n_features = X.shape[1] #number of input neurons
    n_outputs = 1 #number of neurons in the output layer
    batch_size = m

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
        A_list = forward_propagate(X_new, m)
        print('FORWARD PROPAGATION DONE!')

        print('\nBACKWARD PROPAGATION IN ACTION...')
        backpropagation(A_list, Y_new)
        print('BACKPROP DONE!')
    print('NEURAL NETWORK TRAINED!\n')

    test_x = np.loadtxt(
        open("./toy_data/toy_testX.csv", "rb"), delimiter=",", skiprows=0)
    test_y = np.loadtxt(
        open("./toy_data/toy_testY.csv", "rb"),
        delimiter=",",
        skiprows=0,
        dtype=int)

    print('TRAINGING ACCURACY : ' + str(print_accuracy(X, y)))
    print('TESTING ACCURACY : ' + str(print_accuracy(test_x, test_y)))

    #visualising the training decision boundary
    viz.plot_decision_boundary(lambda x: predict(x), X, y)

    #visualising the training decision boundary
    viz.plot_decision_boundary(lambda x: predict(x), test_x, test_y)

    return

if __name__ == '__main__':
    X = np.loadtxt(
        open("./toy_data/toy_trainX.csv", "rb"), delimiter=",", skiprows=0)
    y = np.loadtxt(
        open("./toy_data/toy_trainY.csv", "rb"),
        delimiter=",",
        skiprows=0,
        dtype=int)
    m = len(y)
    y = np.array(y).reshape(m, 1)
    main(X,y)