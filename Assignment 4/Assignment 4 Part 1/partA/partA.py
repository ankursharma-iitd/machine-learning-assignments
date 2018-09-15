from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import os, sys

hash = {} #mappings of string label with int label
max_iterations = 300 #default

def preprocess(X):
    return np.array(X, dtype = np.float32)

def main(num_clusters, X, y, test_X):
    predictions = []
    train_pred = []
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter = max_iterations).fit(X)
    train_labels = kmeans.predict(X)
    test_labels = kmeans.predict(test_X)
    centroid_labels = {}
    for i in range(num_clusters):
        curr_points = np.where(train_labels == i) #points corresponding to a certain label
        corresponding_labels = y[curr_points]
        maj_label = np.argmax(np.bincount(np.array(corresponding_labels)))
        centroid_labels[i] = maj_label
    for i in range(test_X.shape[0]):
        predictions.append(centroid_labels[test_labels[i]])
    for i in range(X.shape[0]):
        train_pred.append(centroid_labels[train_labels[i]])
    return predictions,train_pred

if __name__ == '__main__':
    train_directory = './train'
    test_directory = './test'
    num_clusters = 20
    train_y = []
    train_X = []
    test_X = []
    flag = 0
    label = 0
    for filename in os.listdir(train_directory):
        if filename.endswith(".npy"):
            path_to_file = os.path.join(train_directory, filename)
            X = np.load(path_to_file)
            y = os.path.splitext(filename)[0]
            if(flag == 0):
                train_X = X
                flag = 1
            else:
                train_X = np.vstack((train_X, X))
            for i in range(X.shape[0]):
                train_y.append(label)
            hash[label] = y
            label += 1
    test_X = np.load(os.path.join(test_directory, 'test.npy'))
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)

    X = (train_X * 1.0) / 255.0
    test_X = (test_X * 1.0) / 255.0

    # all_predictions = []
    # shuffle_times = 10
    # for i in range(shuffle_times): #do multiple predictions and then take the maximum
    #     all_predictions.append(main(num_clusters, train_X, np.array(train_y), test_X))

    # all_predictions = np.array(all_predictions).reshape(shuffle_times, test_X.shape[0])

    # final_predictions = []
    # for i in range(test_X.shape[0]):
    #     all_poss = all_predictions[:, i]
    #     maj_voting = np.argmax(np.bincount(all_poss))
    #     final_predictions.append(maj_voting)

    final_predictions,train_predictions =  main(num_clusters, X, np.array(train_y), test_X)
    print('TRAINING SET ACCURACY : ' + str(accuracy_score(train_y, train_predictions)))

    print "ID,CATEGORY"
    for i in range(len(final_predictions)):
        print(str(i) + "," + hash[final_predictions[i]])
