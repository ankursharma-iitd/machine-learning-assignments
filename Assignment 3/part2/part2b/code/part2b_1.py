import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import visualization as viz

def main():
    train_X = np.loadtxt(
        open("./toy_data/toy_trainX.csv", "rb"), delimiter=",", skiprows=0)
    train_Y = np.loadtxt(
        open("./toy_data/toy_trainY.csv", "rb"),
        delimiter=",",
        skiprows=0,
        dtype=int)
    test_X = np.loadtxt(
        open("./toy_data/toy_testX.csv", "rb"), delimiter=",", skiprows=0)
    test_Y = np.loadtxt(
        open("./toy_data/toy_testY.csv", "rb"),
        delimiter=",",
        skiprows=0,
        dtype=int)

    model = linear_model.LogisticRegression()
    model.fit(train_X, train_Y)
    te_y_predict = model.predict(test_X)
    test_acc = accuracy_score(test_Y, te_y_predict)

    tr_y_predict = model.predict(train_X)
    train_acc = accuracy_score(train_Y, tr_y_predict)

    print('TRAINING SET ACCURACY : ' + str(train_acc))
    print('TESTING SET ACCURACY : ' + str(test_acc))

    viz.plot_decision_boundary(lambda i: model.predict(i), train_X, train_Y)
    viz.plot_decision_boundary(lambda i: model.predict(i), test_X, test_Y)
    return

if __name__ == '__main__':
    main()