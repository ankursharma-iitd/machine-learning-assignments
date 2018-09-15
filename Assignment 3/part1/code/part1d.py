from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score

train_data = None
test_data = None
valid_data = None
tr_x = None
tr_y = None
te_x = None
te_y = None
va_x = None
va_y = None

def separate_labels(data):
    x_data = []
    y_data = []
    for somerow in data:
        y_data.append(somerow[0])  #tells you whether rich or not rich
        x_data.append(somerow[1:])
    return np.array(x_data),np.array(y_data)

def main(max_depth, min_samples_leaf_parameter, min_samples_split_parameter):
    model = tree.DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth_parameter,
        min_samples_leaf=min_samples_leaf_parameter,
        min_samples_split=min_samples_split_parameter)
    model.fit(tr_x, tr_y) #this will train the model
    return model

if __name__ == '__main__':
    # import read_data
    # train_data, valid_data, test_data,_ = read_data.init()
    import read_data_c
    train_data, valid_data, test_data, _ = read_data_c.init()
    tr_x, tr_y = separate_labels(train_data)
    te_x, te_y = separate_labels(test_data)
    va_x, va_y = separate_labels(valid_data)
    #best set of parameters
    max_depth_parameter = 12
    min_samples_leaf_parameter = 3
    min_samples_split_parameter = 8

    # best_depth = None
    # best_leaf = None
    # best_split = None
    # best_model = None
    # best_accuracy = -1

    # for max_depth_parameter in range(1,15):
    #     for min_samples_leaf_parameter in range(1, 15):
    #         for min_samples_split_parameter in range(2, 15):
    #             print('RUNNING FOR : ' + str(max_depth_parameter) + ',' + str(min_samples_leaf_parameter) + ',' + str(min_samples_split_parameter))
    #             model = main(max_depth_parameter, min_samples_leaf_parameter, min_samples_split_parameter)
    #             va_y_predict = model.predict(va_x)
    #             valid_acc = accuracy_score(va_y, va_y_predict)
    #             if(valid_acc > best_accuracy):
    #                 print('IMPROVED')
    #                 best_accuracy = valid_acc
    #                 best_model = model
    #                 best_depth = max_depth_parameter
    #                 best_leaf = min_samples_leaf_parameter
    #                 best_split = min_samples_split_parameter
    # model = best_model
    # print('BEST DEPTH : ' + str(best_depth))
    # print('BEST LEAF : ' + str(best_leaf))
    # print('BEST SPLIT : ' + str(best_split))

    model = main(max_depth_parameter, min_samples_leaf_parameter, min_samples_split_parameter)
    va_y_predict = model.predict(va_x)
    valid_acc = accuracy_score(va_y, va_y_predict)

    va_y_predict = model.predict(va_x)
    valid_acc = accuracy_score(va_y, va_y_predict)

    te_y_predict = model.predict(te_x)
    test_acc = accuracy_score(te_y, te_y_predict)

    tr_y_predict = model.predict(tr_x)
    train_acc = accuracy_score(tr_y, tr_y_predict)

    print('TRAINING SET ACCURACY : ' + str(train_acc))
    print('VALIDATION SET ACCURACY : ' + str(valid_acc))
    print('TESTING SET ACCURACY : ' + str(test_acc))