import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm


def compute_cost(X, y, theta):
    J = 0
    A = np.subtract(np.dot(X, theta), y)
    J = (0.5) * (np.dot(A.T, A))
    return J[0, 0]


def gradient_descent(X, y, theta, alpha, epsilon, flag):
    J_hist = []
    all_thetas0 = []
    all_thetas1 = []
    bt_val = 0
    while (1):
        gradient = alpha * np.dot(X.T, np.subtract(np.dot(X, theta), y))
        cost_curr = compute_cost(X, y, theta)
        # print(cost_curr)
        J_hist.append(cost_curr)
        all_thetas0.append(theta[0])
        all_thetas1.append(theta[1])
        if (len(J_hist) >= 2):
            if (math.fabs(J_hist[len(J_hist) - 1] - J_hist[len(J_hist) - 2]) <
                    epsilon):
                break
        if (len(J_hist) >= 2):
            if (math.fabs(J_hist[len(J_hist) - 1] - J_hist[len(J_hist) - 2]) >
                    9999):
                bt_val = 1
                break
        theta = theta - gradient
    if (flag == 1):
        plt.figure(2)
        plt.plot(J_hist)
        plt.xlabel('Number of Iterations -->')
        plt.ylabel('Value of the Cost function(J) -->')
        plt.title('Checking convergence with alpha : ' + str(float(alpha)))
        plt.show(block=False)
        raw_input("Hit Enter To Close")
        plt.close()
    return theta, np.array(all_thetas0), np.array(all_thetas1), np.array(
        J_hist), bt_val


def main(x_training, y_training):
    x_training = x_training.T
    y_training = y_training.T

    #plot the initial training data
    print('\n\nPlotting the data...\n')
    plt.figure(1)
    plt.plot(x_training, y_training, 'ro')
    plt.xlabel('Acidity -->')
    plt.ylabel('Density -->')
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
    theta = np.reshape(np.zeros(2), (2, 1))  #0-initialization of theta

    #some gradient descent settings
    epsilon = math.pow(10, -10)
    alpha = 0.0007

    print('\nTesting the cost function ...\n')
    #compute and display the initial cost
    J = compute_cost(X, y_training, theta)
    print('With theta = [0 ; 0]\nCost computed = ' + str(J) + '\n')

    #run gradient descent
    print('\n Running Gradient Descent Algorithm... \n')
    # print(theta)
    theta, thetas0, thetas1, all_cost, _ = gradient_descent(
        X, y_training, theta, alpha, epsilon, 1)
    opti0 = theta[0, 0]
    opti1 = theta[1, 0]
    print("Optimal Value of Theta-0" + str(opti0))
    print("Optimal Value of Theta-0" + str(opti1))

    #plotting the predicted line learnt from linear regression
    plt.figure(3)
    plt.plot(x_training, y_training, 'ro', label='Input Example', markersize=5)
    plt.xlabel('Acidity -->')
    plt.ylabel('Density -->')
    plt.title('Prediction Model using Linear Regression')
    plt.plot(x_training, np.dot(X, theta), label='Prediction Line')
    plt.legend()
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()

    #Visualizing J(theta_0, theta_1) as a 3D-mesh followed from one of the quesions on stackoverflow
    print('Visualizing J(theta_0, theta_1) ...\n')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ms = np.linspace(-0.2, 2.0, 100)
    bs = np.linspace(-1, 1, 100)
    M, B = np.meshgrid(ms, bs)
    zs = np.array([
        compute_cost(X, y_training, np.array([[mp], [bp]]))
        for mp, bp in zip(np.ravel(M), np.ravel(B))
    ])
    Z = zs.reshape(M.shape)
    surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    blue_patch = mpatches.Patch(color='blue', label='Cost Function')
    red_patch = mpatches.Patch(color='red', label='Path taken by GD')
    ax.plot(thetas0, thetas1, all_cost, color='r', alpha=0.5)
    ax.set_xlabel('theta(0)')
    ax.set_ylabel('theta(1)')
    ax.set_zlabel('J(theta)')
    plt.legend(loc='upper left', handles=[blue_patch, red_patch])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('J(theta) vs theta')
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()

    #Plotting the contour
    print('\nPlotting the contour ... \n')
    fig = plt.figure()
    CS = plt.contour(M, B, Z, 25)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot([opti0], [opti1], color='r', marker='x', label='Optimal Value')
    plt.legend()
    plt.xlabel('Theta0 -->')
    plt.ylabel('Theta1 -->')
    plt.title('Contour plot')
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()

    #plotting the GD animation using animation
    print('\n3D Animation of GD Convergence ...\n')
    plot_3D(X, y_training, thetas0, thetas1, all_cost, opti0, opti1)

    #plotting the contour animation using looping
    print('\nContour Animation of GD Convergence ...\n')
    plot_contour(X, y_training, alpha, epsilon)
    plt.close()

    #plotting for other values of eta
    eta = [0.001, 0.005, 0.009, 0.013, 0.017, 0.021, 0.025]
    check = 0
    for some_alpha in eta:
        if (check == 0):
            print('\nContour Animation of GD Convergence with eta = ' +
                  str(some_alpha) + '\n')
            bt_val = plot_contour(X, y_training, some_alpha, epsilon)
            if (bt_val == 1):
                check = 1
                print('Gradient Descent doesn\'t converge for eta : ' +
                      str(some_alpha) + '\n')
        else:
            print('Gradient Descent doesn\'t converge for eta : ' +
                  str(some_alpha) + '\n')
    return


def plot_contour(X, y_training, alpha, epsilon):
    theta = np.reshape(np.zeros(2), (2, 1))  #0-initialization of theta
    theta, thetas0, thetas1, all_cost, bt_val = gradient_descent(
        X, y_training, theta, alpha, epsilon, 0)
    if (bt_val == 1):
        return bt_val
    opti0 = theta[0, 0]
    opti1 = theta[1, 0]

    ms = np.linspace(-0.2, 2.0, 100)
    bs = np.linspace(-1, 1, 100)
    M, B = np.meshgrid(ms, bs)
    zs = np.array([
        compute_cost(X, y_training, np.array([[mp], [bp]]))
        for mp, bp in zip(np.ravel(M), np.ravel(B))
    ])
    Z = zs.reshape(M.shape)
    fig = plt.figure()
    CS = plt.contour(M, B, Z, 25)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot([opti0], [opti1], color='r', marker='x', label='Optimal Value')
    plt.legend()
    plt.xlabel('Theta0 -->')
    plt.ylabel('Theta1 -->')
    plt.title('Contour plot with Learning Rate : ' + str(alpha))
    for i in range(0, len(all_cost), 2):
        plt.plot(thetas0[:i], thetas1[:i], color='r')
        plt.draw()
        plt.pause(0.05)
        print('Iteration : ' + str(i))
    print('It converges in ' + str(len(all_cost)) + ' iterations\n')
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    return 0


def plot_3D(X, y_training, thetas0, thetas1, all_cost, opti0, opti1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ms = np.linspace(-0.2, 2.0, 100)
    bs = np.linspace(-1, 1, 100)
    M, B = np.meshgrid(ms, bs)
    zs = np.array([
        compute_cost(X, y_training, np.array([[mp], [bp]]))
        for mp, bp in zip(np.ravel(M), np.ravel(B))
    ])
    Z = zs.reshape(M.shape)
    ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.5)
    blue_patch = mpatches.Patch(color='blue', label='Cost Function')
    red_patch = mpatches.Patch(color='red', label='Path taken by GD')
    ax.set_xlabel('theta(0)')
    ax.set_ylabel('theta(1)')
    ax.set_zlabel('J(theta)')
    plt.legend(loc='upper left', handles=[blue_patch, red_patch])
    ax.set_title('J(theta) vs theta')
    theta0_new = []
    theta1_new = []
    cost_new = []
    j = 0
    num_iterations = len(all_cost)

    #one way by using animation, issues setting the interval time
    def init():
        ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.5)
        plt.plot(
            [opti0], [opti1], color='r', marker='x', label='Optimal Value')
        return

    def update(i, step_size):
        j = (step_size * i)
        theta0_new.append(thetas0[j])
        theta1_new.append(thetas1[j])
        cost_new.append(all_cost[j])
        ax.plot(theta0_new, theta1_new, cost_new, color='r')
        plt.draw()
        print('Iteration : ' + str(j) + ', Error Value : ' + str(all_cost[j]))
        if (i == ((num_iterations - 1) / step_size - 1)):
            print('\nConverged Successfully\n')
        return

    step_size = 2
    ani = FuncAnimation(
        fig,
        update,
        frames=(num_iterations - 1) / step_size,
        fargs=(step_size, ),
        init_func=init,
        interval=1,
        repeat=False)

    # for i in range(0, num_iterations, step_size):
    #     ax.plot(thetas0[:i], thetas0[:i], all_cost[:i], color='r')
    #     plt.draw()
    #     plt.pause(0.0005)
    #     print('Iteration : ' + str(i))

    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()


if __name__ == '__main__':
    x_training = []  #denotes the input training set
    y_training = []  #denotes the outpur training set
    with open('./Assignment_1_datasets/linearX.csv') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            x_training.append(row[0])
    with open('./Assignment_1_datasets/linearY.csv') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            y_training.append(row[0])
    main(
        np.array([x_training]).astype(np.float),
        np.array([y_training]).astype(np.float))
