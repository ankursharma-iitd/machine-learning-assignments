import math
import numpy as np
import matplotlib.pyplot as plt
import csv

global flag
flag = 0

def get_phi(label, m):
    return len(label)*1.0/m

def get_mu(label, X, m):
    sum = 0
    for i in label:
        sum += X[i]
    return sum*1.0/len(label)

def get_indiv_sigma(label, X, mu, m):
    mu = mu.reshape(len(mu),1)
    sum = 0
    for i in label:
        x_i = X[i].reshape(len(X[i]),1)
        some_i = x_i - mu
        sum += np.dot(some_i, some_i.T)
    return sum*1.0/len(label)

def same_sigma(X,y,mu0,mu1,m):
    mu0 = mu0.reshape(len(mu0), 1)
    mu1 = mu1.reshape(len(mu1), 1)
    sum = 0
    for i in range(m):
        x_i = X[i].reshape(len(X[i]), 1)
        if(y[i] == 0):
            some_i = x_i - mu0
        else:
            some_i = x_i - mu1
        sum += np.dot(some_i, some_i.T)
    return sum*1.0/m

def get_y(x, c, a, b):
    return (c - a*x*1.0)/b

def get_quadratic(x, phi, mu0, mu1, sigma0, sigma1):
    global flag
    sigma0_inv = np.linalg.pinv(sigma0)
    sigma1_inv = np.linalg.pinv(sigma1)
    a = sigma1_inv[0][0]
    b = sigma1_inv[0][1]
    c = sigma1_inv[1][0]
    d = sigma1_inv[1][1]
    p = sigma0_inv[0][0]
    q = sigma0_inv[0][1]
    r = sigma0_inv[1][0]
    s = sigma0_inv[1][1]
    p1 = phi
    p0 = 1.0 - phi
    C = np.log(np.linalg.det(sigma0)) - np.log(np.linalg.det(sigma1)) + 2.0*np.log(p1) - 2.0*np.log(p0)
    u = d - s
    v = (-2.0 * d * mu1[1]) + (2.0 * s * mu0[1]) + (b * x) - (b * mu1[0]) + (c * x) - (c * mu1[0]) - (q * x) + (q * mu0[0]) - (r * x) + (r * mu0[0])
    w = C - (a * ((x - mu1[0])**2)) + (p * ((x - mu0[0])**2)) + (mu0[1] * (q + r) * (mu0[0] - x)) + (mu1[1] * (b + c) * (x - mu1[0])) - (d*(mu1[1] ** 2)) + (s*(mu0[1]**2))
    sqt = np.sqrt((v**2) + (4*u*w))
    if(flag == 0):
        A = a - p
        B = b + c - q - r
        C = d - s
        D = -2*a*mu1[0] - mu1[1]*(b + c) + 2*p*mu0[0] + mu0[1]*(q + r)
        E = (-2.0 * d * mu1[1]) + (2.0 * s * mu0[1]) - (b * mu1[0]) - (
            c * mu1[0]) + (q * mu0[0]) + (r * mu0[0])
        F = -C + (a * (mu1[0])**2) - (p * ((mu0[0])**2)) - (
            mu0[1] * (q + r) *
            (mu0[0])) + (mu1[1] * (b + c) *
                            (mu1[0])) + (d * (mu1[1]**2)) - (s * (mu0[1]**2))
        print('The value of discriminant is ' + str(B**2 - 4*A*C))
        print('The equation of decision boundary is a conic of the form : ')
        print(str(A) + 'x^2 + ' + str(B) + 'xy + ' + str(C) + 'y^2 + ' + str(D) + 'x + ' + str(E) + 'y + ' + str(F) + '\n')
        flag = 1
    return ( (0.5*(-v + sqt))/u , (0.5*(-v - sqt))/u )


def main(X, y, m):
    label_0 = np.where(y == 0)[0]
    label_1 = np.where(y == 1)[0]

    #plotting the training data with the linear separator
    print('\n\nPlotting the data...\n')
    plt.figure()
    x1 = np.array([X[x, :] for x in label_0])
    x2 = np.array([X[x, :] for x in label_1])
    plt.plot(x1[:, 0], x1[:, 1], 'ro', marker='o', label='Alaska')
    plt.plot(x2[:, 0], x2[:, 1], 'bo', marker='^', label='Canada')
    plt.xlabel('x1 -->')
    plt.ylabel('x2 -->')
    plt.title('Training Set')
    plt.legend()
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()

    #describing the result for part (a)
    phi = get_phi(label_1, m)
    print('Bernoulli phi : ' + str(phi))
    print('Mean Vector 0 : ')
    mu0 = get_mu(label_0, X, m)
    print(mu0)
    print('Mean Vector 1 : ')
    mu1 = get_mu(label_1, X, m)
    print(mu1)
    print('Covariance Matrix : ')
    sigma = same_sigma(X, y, mu0, mu1, m)
    print(sigma)

    #plotting the training data with the linear separator
    print('\n\nPlotting the data...\n')
    plt.figure()
    x1 = np.array([X[x, :] for x in label_0])
    x2 = np.array([X[x, :] for x in label_1])
    plt.plot(x1[:, 0], x1[:, 1], 'ro', marker='o', label='Alaska')
    plt.plot(x2[:, 0], x2[:, 1], 'bo', marker='^', label='Canada')
    plt.xlabel('x1 -->')
    plt.ylabel('x2 -->')
    plt.title('Training Set')
    #calculations
    sigma_inv = np.linalg.pinv(sigma)
    vec_1 = sigma_inv.dot(mu0 - mu1)
    c = 0.5 * (((mu0.T).dot(sigma_inv)).dot(mu0) -
               ((mu1.T).dot(sigma_inv)).dot(mu1))
    x_lines = [np.min(X[:, 0]), np.max(X[:, 0])]
    #plotting the boundary
    plt.plot(
        x_lines, [
            get_y(x_lines[0], c, vec_1[0], vec_1[1]),
            get_y(x_lines[1], c, vec_1[0], vec_1[1])
        ],
        label='Linear Decision Boundary')
    plt.legend()
    plt.show(block=False)
    print('The equation of DB is : ')
    print(str(vec_1[0]) + 'x + ' + str(vec_1[1]) + 'y = ' + str(c))
    raw_input('\nPress Enter to close\n')
    plt.close()

    #describing the result for part (d)
    phi = get_phi(label_1, m)
    print('Bernoulli phi : ' + str(phi))
    print('Mean Vector 0 : ')
    mu0 = get_mu(label_0, X, m)
    print(mu0)
    print('Mean Vector 1 : ')
    mu1 = get_mu(label_1, X, m)
    print(mu1)
    print('Covariance Vector 0 : ')
    sigma0 = get_indiv_sigma(label_0, X, mu0, m)
    print(sigma0)
    print('Covariance Vector 1 : ')
    sigma1 = get_indiv_sigma(label_1, X, mu1, m)
    print(sigma1)

    #plotting the quadratic boundary
    num_points = 1000
    x_all = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), num_points)
    x_plot1 = []
    x_plot2 = []
    y_plot1 = []
    y_plot2 = []
    count = 0
    for i in range(num_points):
        x = x_all[i]
        (y_val1, y_val2) = get_quadratic(x, phi, mu0, mu1, sigma0, sigma1)
        x_plot1.append(x)
        y_plot1.append(y_val1)
        if(count == 10):
            x_plot2.append(x)
            y_plot2.append(y_val2)
            count = 0
        count += 1

    #plotting the training data with the quadratic separator
    print('\n\nPlotting the data...\n')
    plt.figure()
    x1 = np.array([X[x, :] for x in label_0])
    x2 = np.array([X[x, :] for x in label_1])
    plt.plot(x1[:, 0], x1[:, 1], 'ro', marker='o', label='Alaska')
    plt.plot(x2[:, 0], x2[:, 1], 'bo', marker='^', label='Canada')
    plt.xlabel('x1 -->')
    plt.ylabel('x2 -->')
    plt.title('Training Set')
    #plotting the boundary
    plt.plot(x_plot1,y_plot1,'ko',markersize = 1, label='Quadratic Decision Boundary')
    plt.plot(
        x_plot2,
        y_plot2,
        'ko',
        markersize=0.5)
    plt.plot(
    x_lines, [
        get_y(x_lines[0], c, vec_1[0], vec_1[1]),
        get_y(x_lines[1], c, vec_1[0], vec_1[1])
    ],
    label='Linear Decision Boundary')
    plt.legend()
    plt.show(block=False)
    raw_input('\nPress Enter to close\n')
    plt.close()


if __name__ == '__main__':
    X = np.loadtxt(
        open("./Assignment_1_datasets/q4x.dat", "rb"),
        delimiter="  ",
        skiprows=0)
    labels = ['Alaska', 'Canada'] #list of all the labels
    y = [] #Canada refers to example 1, and Alaska refers to example 0
    with open('./Assignment_1_datasets/q4y.dat') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            if(row[0] == labels[0]):
                y.append(0)
            else:
                y.append(1)
    m = len(y)
    # X = np.hstack((np.reshape(np.ones(m), (m, 1)), X))
    temp = X[:,0].copy()
    temp -= np.mean(X[:, 0])
    temp /= np.std(X[:, 0])
    X[:, 0] = temp
    temp = X[:, 1].copy()
    temp -= np.mean(X[:, 1])
    temp /= np.std(X[:, 1])
    X[:, 1] = temp
    main(X, np.array([y]).reshape(m, 1),m)
