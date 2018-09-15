from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os, sys

num_of_components = 50 #project onto these 50 features
hash = {}

def preprocess(X):
    return np.array(X, dtype = np.float32)

def num_to_label(trainactuals, trainpredictions, inputname, outputname):
    inputlabels = np.loadtxt(inputname)
    trainpredictions = np.loadtxt(trainpredictions)
    f = open(outputname, 'w')
    f.write('TRAINING SET ACCURACY : ' + str(accuracy_score(trainactuals, trainpredictions)) + '\n')
    f.write("ID,CATEGORY\n")
    for i in range(len(inputlabels)):
        f.write(str(i) + "," + hash[inputlabels[i]] + "\n")
    f.close()
    return

def main(num_classes, X, y, test_X):
    pca_train = PCA(n_components=num_of_components)
    pca_train.fit(X)

    train_x_data = pca_train.transform(X)
    test_x_data = pca_train.transform(test_X)

    print('\nPRINTING THE TRAIN CSV FILE...')
    with open('./svm/new_train', 'w') as f:
        for i in range(train_x_data.shape[0]):
            curr_ex = train_x_data[i]
            curr_label = y[i]
            string = str(curr_label) + ' '
            count = 1
            for j in range(len(curr_ex)):
                string += str(count) + ':' + str(curr_ex[j]) + ' '
                count += 1
            string += '\n'
            f.write(string)
        f.close()
    print('TRAIN CSV FILE HAS BEEN CREATED!\n')

    print('\nPRINTING THE TEST CSV FILE...')
    with open('./svm/new_test', 'w') as f:
        for i in range(test_x_data.shape[0]):
            curr_ex = test_x_data[i]
            curr_label = 0
            string = str(curr_label) + ' '
            count = 1
            for j in range(len(curr_ex)):
                string += str(count) + ':' + str(curr_ex[j]) + ' '
                count += 1
            string += '\n'
            f.write(string)
        f.close()
    print('TEST CSV FILE HAS BEEN CREATED!\n')

    return


if __name__ == '__main__':
    train_directory = './train'
    test_directory = './test'
    num_classes = 20
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
    flag = int(sys.argv[1])
    if(flag == 0):
        print "yay"
        test_X = np.load(os.path.join(test_directory, 'test.npy'))
        train_X = preprocess(train_X)
        test_X = preprocess(test_X)

        scaler = StandardScaler()
        scaler.fit(train_X)

        X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)
        main(num_classes, train_X, np.array(train_y), test_X)
    else:
        print "nay"
        num_to_label(np.array(train_y), sys.argv[2], sys.argv[3], sys.argv[4])

    # f = open('pca_svm.csv', 'w')
    # f.write("ID,CATEGORY\n")
    # for i in range(len(final_predictions)):
    #     f.write(str(i) + "," + hash[final_predictions[i]] + "\n")
    # f.close()
