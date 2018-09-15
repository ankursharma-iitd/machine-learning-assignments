from sklearn.ensemble import RandomForestClassifier
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

def main(n_estimators, bootstrap, max_features):
    model = RandomForestClassifier(
        criterion="entropy",
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=bootstrap)
    model = model.fit(tr_x, tr_y) #this will train the model
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
    n_estimators = 19
    bootstrap = True
    max_features = 'log2'

    # best_n_estimators = None
    # best_bootstrap = None
    # best_max_features = None
    # best_model = None
    # best_accuracy = -1

    # for n_estimators in range(1,30):
    #     for bootstrap in [True, False]:
    #         for max_features in ['auto', 'log2', 'sqrt']:
    #             print('RUNNING FOR : ' + str(n_estimators) + ',' + str(bootstrap) + ',' + str(max_features))
    #             model = main(n_estimators, bootstrap, max_features)
    #             va_y_predict = model.predict(va_x)
    #             valid_acc = accuracy_score(va_y, va_y_predict)
    #             if(valid_acc > best_accuracy):
    #                 print('IMPROVED')
    #                 best_accuracy = valid_acc
    #                 best_model = model
    #                 best_n_estimators = n_estimators
    #                 best_bootstrap = bootstrap
    #                 best_max_features = max_features
    # model = best_model
    # print('BEST N-ESTIMATOR : ' + str(best_n_estimators))
    # print('BEST BOOTSTRAP : ' + str(best_bootstrap))
    # print('BEST MAX FEATURES : ' + str(best_max_features))

    model = main(n_estimators, bootstrap, max_features)
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