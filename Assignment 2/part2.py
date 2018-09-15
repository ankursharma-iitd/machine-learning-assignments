import sys
import pickle
from collections import Counter
import operator
import numpy as np
from svmutil import *

model_number  = int(sys.argv[1])
input_name = sys.argv[2]
output_name = sys.argv[3]
corresponding_model = 'models_2/model' + sys.argv[1] + '.p'
test_x_data = []
num_examples = 0

#predict a class from the classifiers learnt
def get_predicted_class(x, w_classifiers, b_classifiers):
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

def get_total_prediction(all_examples, w_classifiers, b_classifiers):
    y = []
    for each_example in all_examples:
        y.append(get_predicted_class(each_example, w_classifiers, b_classifiers))
    return y


def scale_features(data):
    #scale all your features in the range [0, 1] by (x - xmin)/(xmax - xmin) for every feature x
    A = np.array(data)
    return ((A * 1.0) / 255.0)


def get_accuracy(prediction, actual):
    length = len(prediction)
    sum = 0.0
    for i in range(length):
        if (int(prediction[i]) == int(actual[i])):
            sum += 1.0
    return ((sum * 100.0) / length)


if __name__ == '__main__':
    check_flag = 0
    test_y_data = []
    print('\nREADING THE TEST CSV FILE...')
    with open(input_name) as f:
        content = f.readlines()
        for x in content:
            pixels = list(map(int, x.split(',')))
            num_features = len(pixels)
            if(check_flag == 1):
                num_features = len(pixels) - 1
                test_y_data.append(pixels[num_features])
            else:
                num_features = len(pixels)
                test_y_data.append(0)
            test_x_data.append(pixels[:num_features])
            num_examples += 1
    print('TEST CSV FILE HAS BEEN READ!\n')

    if(model_number == 1):
        #use the pegasos model for making the predictions
        file = open(corresponding_model, 'rb')
        w_classifiers = pickle.load(file)
        b_classifiers = pickle.load(file)
        num_examples_test = len(test_x_data)

        print('\nSCALING FEATURES FOR TEST CSV FILE IN THE RANGE [0, 1]...')
        test_x_data = scale_features(test_x_data)
        print('FEATURES SCALED!\n')

        print('\nTEST SET PREDICTION...')
        test_prediction = get_total_prediction(test_x_data, w_classifiers, b_classifiers)
        print('DONE!\n')

        test_accuracy = get_accuracy(test_prediction, test_y_data)
        print(str(test_accuracy))

        with open(output_name, 'w+') as f:
            for each_prediction in test_prediction:
                string = str(each_prediction) + '\n'
                f.write(string)
            f.close()

    elif(model_number == 2):
        #use the linear kernel model for making the prediction
        output_train_csv = output_name
        print('\nPRINTING THE NEW CSV FILE...')
        with open(output_train_csv, 'w+') as f:
            for i in range(num_examples):
                curr_ex = test_x_data[i]
                curr_y = test_y_data[i]
                string = str(curr_y) + ' '
                count = 1
                for j in range(len(curr_ex)):
                    string += str(count) + ':' + str(curr_ex[j]) + ' '
                    count += 1
                string += '\n'
                f.write(string)
            f.close()
        print('CSV FILE HAS BEEN CREATED!\n')
    elif(model_number == 3):
        #use the best libsvm model that you have
        output_train_csv = output_name
        print('\nPRINTING THE NEW CSV FILE...')
        with open(output_train_csv, 'w+') as f:
            for i in range(num_examples):
                curr_ex = test_x_data[i]
                curr_y = test_y_data[i]
                string = str(curr_y) + ' '
                count = 1
                for j in range(len(curr_ex)):
                    string += str(count) + ':' + str(curr_ex[j]) + ' '
                    count += 1
                string += '\n'
                f.write(string)
            f.close()
        print('CSV FILE HAS BEEN CREATED!\n')
    else:
        print('INCORRECT MODEL NUMBER. EXITING...')
        exit(0)