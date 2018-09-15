import os
import re
import numpy as np
import random
from collections import Counter
import operator
import pickle
import time

train_x_data = [] #input training set
train_y_date = [] #labels for the training set
test_x_data = [] #input test set
test_y_data = [] #labels for the test set
num_examples = 0
num_examples_test = 0
num_features = 0
examples_per_class = {}  #store the examples relevant to each class as a hashtbale
num_classes = 0
num_classifiers = 0
w_classifiers = {}
b_classifiers = {}
epsilon = 10 ** -6

#convergence when difference in the w values in consecutive iterartions becomes less than a certain epsilon
def convergence_check(old_w, new_w):
    diff = np.max(np.abs(new_w - old_w))
    if(diff < epsilon):
        return True
    # print('DIFFERENCE : ' + str(diff))
    return False

def get_accuracy(prediction, actual):
    length = len(prediction)
    sum = 0.0
    for i in range(length):
        if (int(prediction[i]) == int(actual[i])):
            sum += 1.0
    return ((sum * 100.0) / length)

def run_minibatch_pegasos(minusone, plusone):
    #initialising the parameters
    left = len(minusone)
    right = len(plusone)
    all_examples = minusone + plusone
    number_corre_examples = len(all_examples)
    w = np.array([0.0 for i in range(num_features)]).reshape(num_features, 1)
    b = 0.0
    iters = 1
    C = 1.0
    batch_size = 100 #stepsize for the algorithm
    max_iterations = 2000 #set the maximum number of iterations
    lambbda = (1.0 / (number_corre_examples * C * 1.0))

    while(True):
        # print('RUNNING FOR ITERATION : ' + str(iters))
        old_w = w
        old_b = b
        eta_t = (1.0 / (1.0 * lambbda * iters))
        A_t = random.sample(xrange(number_corre_examples), batch_size) #randomly sample some k examples
        some_A_t = [all_examples[i] for i in A_t]
        random_matrix_x = np.array([train_x_data[i] for i in some_A_t]) #get the random matrix of examples
        randomly_y = []
        for i in A_t:
            if(i < left):
                randomly_y.append(-1)
            else:
                randomly_y.append(1)
        random_matrix_y = np.array(randomly_y).reshape(batch_size, 1) #get the random matrix of corresponding labels
        something = np.multiply(random_matrix_y, np.dot(random_matrix_x, old_w) + old_b)
        A_t_plus = np.where(something < 1)[0]
        x_matrix_A_t_plus = np.array([random_matrix_x[i] for i in A_t_plus]).reshape(len(A_t_plus), num_features)
        y_matrix_A_t_plus = np.array([random_matrix_y[i] for i in A_t_plus]).reshape(len(A_t_plus), 1)
        something = np.sum(np.multiply(x_matrix_A_t_plus, y_matrix_A_t_plus), axis = 0).reshape(num_features, 1)
        #update the value of w
        w = ((1.0 - (eta_t * lambbda * 1.0)) * old_w) + (((eta_t * 1.0) / batch_size) * something)
        #update the value of b
        b = old_b + (((eta_t * 1.0) * C) * np.sum(y_matrix_A_t_plus))
        #convergence check below
        if ((iters == max_iterations) or (convergence_check(np.append(old_w, [old_b]), np.append(w, [b])))):
            break
        iters += 1
    return w,b

#predict a class from the classifiers learnt
def get_predicted_class(x):
    all_classifications = []
    minusclasses = w_classifiers.keys()
    for i in range(len(minusclasses)):
        class_i = minusclasses[i]
        plusclasses = w_classifiers[class_i].keys()
        for j in range(len(plusclasses)):
            class_j = plusclasses[j]
            w = w_classifiers[class_i][class_j].reshape(num_features, 1)
            b = b_classifiers[class_i][class_j]
            x = np.array(x).reshape(num_features, 1)
            val = np.dot(w.T, x) + b
            if(val >= 0):
                all_classifications.append(class_j)
            else:
                all_classifications.append(class_i)
    #output the one with the highest wins
    hashed_table = Counter(all_classifications)
    sorted_x_1 = sorted(hashed_table.items(), key=operator.itemgetter(0))
    max_count = float("-inf")
    pred_class = '0'
    for key,value in sorted_x_1:
        if(value >= max_count):
            max_count = value
            pred_class = key
    return pred_class

def get_total_prediction(all_examples):
    y = []
    for each_example in all_examples:
        y.append(get_predicted_class(each_example))
    return y

def learn_classifiers():
    classes = examples_per_class.keys()
    count = 1
    #classs one -> -1
    #class two -> +1
    for i in range(num_classes):
        class_one = classes[i]
        w_hash = {}
        b_hash = {}
        for j in range(num_classes):
            class_two = classes[j]
            if((i == j) or (class_two in w_classifiers.keys() and class_one in w_classifiers[class_two].keys())):
                continue
            exam_one = examples_per_class[class_one]
            exam_two = examples_per_class[class_two]
            print('LEARNING THE CLASSIFIER : ' + str(count))
            w,b = run_minibatch_pegasos(exam_one, exam_two)
            w_hash[class_two] = w
            b_hash[class_two] = b
            count += 1
        w_classifiers[class_one] = w_hash
        b_classifiers[class_one] = b_hash
    return

def scale_features(data):
    #scale all your features in the range [0, 1] by (x - xmin)/(xmax - xmin) for every feature x
    A = np.array(data)
    return ((A * 1.0) / 255.0)

if __name__ == '__main__':
    train_csv = 'mnist/train.csv'
    print('\nREADING THE TRAIN CSV FILE...')
    with open(train_csv) as f:
        content = f.readlines()
        for i in range(len(content)):
            x = content[i]
            pixels = list(map(int, x.split(',')))
            num_features = len(pixels) - 1
            y = pixels[num_features]
            if not y in examples_per_class.keys():
                examples_per_class[y] = [i]
            else:
                examples_per_class[y].append(i);
            train_y_date.append(y)
            train_x_data.append(pixels[:num_features])
    print('TRAIN CSV FILE HAS BEEN READ!\n')
    num_examples = len(train_x_data)
    num_classes = len(examples_per_class.keys())
    num_classifiers = (num_classes * (num_classes - 1)) / 2

    print('\nSCALING FEATURES FOR TRAIN CSV FILE IN THE RANGE [0, 1]...')
    train_x_data = scale_features(train_x_data)
    print('FEATURES SCALED!\n')

    test_csv = 'mnist/test.csv'
    print('\nREADING THE TEST CSV FILE...')
    with open(test_csv) as f:
        content = f.readlines()
        for x in content:
            pixels = list(map(int, x.split(',')))
            test_y_data.append(pixels[num_features])
            test_x_data.append(pixels[:num_features])
    print('TEST CSV FILE HAS BEEN READ!\n')
    num_examples_test = len(test_x_data)

    print('\nSCALING FEATURES FOR TEST CSV FILE IN THE RANGE [0, 1]...')
    test_x_data = scale_features(test_x_data)
    print('FEATURES SCALED!\n')

    print('\nLEARNING THE CLASSIFIERS...\n')
    time_start = time.clock()
    learn_classifiers()
    time_45_classifiers = (time.clock() - time_start)
    print('CLASSIFIERS LEARNT!\n')

    #predict over the set heh
    print('\nTRAINING SET PREDICTION...')
    time_start = time.clock()
    train_prediction = get_total_prediction(train_x_data)
    time_training_prediction = time.clock() - time_start

    print('\nTEST SET PREDICTION...')
    time_start = time.clock()
    test_prediction = get_total_prediction(test_x_data)
    time_test_prediction = time.clock() - time_start

    #compute accuracy for the SVM
    train_accuracy = get_accuracy(train_prediction, train_y_date)
    test_accuracy = get_accuracy(test_prediction, test_y_data)

    print('\nACCURACIES OBTAINED FROM MULTICLASS SVM CLASSIFIER : ')
    print('TRAINING SET ACCURACY : ' + str(train_accuracy))
    print('TEST SET ACCURACY : ' + str(test_accuracy))

    print('\nTIMING DETAILS : ')
    print('TIME TAKEN TO LEARN ' + str(num_classifiers) + ' CLASSIFIERS : ' + str(time_45_classifiers))
    print('TIME TAKEN FOR TRAINING SET PREDICTION : ' + str(time_training_prediction))
    print('TIME TAKEN FOR TEST SET PREDICTION : ' + str(time_test_prediction))

    #save the model as an external file
    file = open('pegasos_model.p', 'wb')
    pickle.dump(w_classifiers, file)
    pickle.dump(b_classifiers, file)
    file.close()
