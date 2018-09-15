import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import time


def normal_equation_linear(X, Y):
    return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))


def main(x_training, y_training):
    x_training = x_training.T
    y_training = y_training.T

    #plot the initial training data
    print('\n\nPlotting the data...\n')
    plt.figure(1)
    plt.plot(x_training, y_training, 'ro')
    plt.xlabel('Input X -->')
    plt.ylabel('Input Y -->')
    plt.title('Training Set')
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()

    #Design matrix X

    x_training = (x_training - np.mean(x_training)) / (
        np.std(x_training))  #feature scaling using mean normalization

    # x_training = (x_training - np.mean(x_training)) / (np.max(x_training) - np.mean(x_training))
    m = y_training.shape[0]  #number of training examples
    X = np.hstack((np.reshape(np.ones(m), (m, 1)), x_training))
    theta = normal_equation_linear(X, y_training)

    #plot the prediction line
    print('\n\nPlotting the data...\n')
    plt.figure(1)
    plt.plot(x_training, y_training, 'ro', label='Input Example', markersize=5)
    plt.xlabel('Input X -->')
    plt.ylabel('Input Y -->')
    plt.title('Linear Regression Model using Normal Equations')
    plt.plot(x_training, np.dot(X, theta), label='Prediction Line')
    plt.legend()
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()

    #get the weights using the exponential function with deviation  = 0.8
    sigma = 0.8
    x_min = np.min(x_training)
    x_max = np.max(x_training)
    num_of_points = 20
    x_values = np.linspace(x_min, x_max, num_of_points)
    plot_for_tau(X, x_training, y_training, sigma, num_of_points, x_values)
    plt.close()

    #plotting for varous values of tau
    tau_list = [0.1, 0.3, 2, 10]
    for tau in tau_list:
        print('Plotting for Tau : ' + str(tau))
        plot_for_tau(X, x_training, y_training, tau, num_of_points,x_values)
    return


def plot_for_tau(X, x_training, y_training, sigma, num_of_points, x_values):
    m = len(x_training)
    list_of_prediction = []
    for x in x_values:
        x_array = x * (np.ones(m).reshape(m, 1))
        weight_array = np.exp(
            np.multiply(-1,
                        np.square(np.subtract(x_array, x_training)) / float(
                            2 * sigma * sigma)))
        W = np.diag(weight_array.reshape(m))
        theta = np.dot(
            np.linalg.pinv(np.dot(X.T, np.dot(W, X))),
            np.dot(X.T, np.dot(W, y_training)))
        local_prediction = theta[0] + (theta[1] * x)
        list_of_prediction.append(local_prediction)

    #plot the prediction line
    print('\n\nPlotting the data...\n')
    plt.figure()
    plt.plot(x_training, y_training, 'ro', label='Input Example', markersize=5)
    plt.xlabel('Input X -->')
    plt.ylabel('Input Y -->')
    plt.title(
        'Locally Weighted Linear Regression Model using Tau : ' + str(sigma))
    plt.plot(x_values, list_of_prediction, label='Prediction Curve')
    plt.legend()
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    return


if __name__ == '__main__':
    x_training = []  #denotes the input training set
    y_training = []  #denotes the outpur training set
    with open('./Assignment_1_datasets/weightedX.csv') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            x_training.append(row[0])
    with open('./Assignment_1_datasets/weightedY.csv') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            y_training.append(row[0])
    main(
        np.array([x_training]).astype(np.float),
        np.array([y_training]).astype(np.float))