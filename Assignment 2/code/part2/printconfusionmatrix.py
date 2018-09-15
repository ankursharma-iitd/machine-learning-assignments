import sys
import matplotlib.pyplot as plt
import numpy as np

examples_per_class_actual = {}
num_classes = 0

def print_confusion_matrix(examples_per_class_actual, predicted_labels):
    confusion_matrix = [[0 for i in range(num_classes)]
                        for j in range(num_classes)]
    confusion_matrix = np.array(confusion_matrix)
    classes = examples_per_class_actual.keys()
    for i in range(num_classes):
        each_class = classes[i]
        corresponding_examples = examples_per_class_actual[each_class]
        for each_example in corresponding_examples:
            predicted_class = predicted_labels[each_example]
            j = classes.index(predicted_class)
            confusion_matrix[j][i] += 1

    print('\t\tA  C  T  U  A  L   \t\tC  L  A  S  S  E  S')
    class_new = [str(classi) for classi in classes]
    print('\t' + '\t'.join(class_new))
    for i in range(num_classes):
        print str(classes[i]) + '\t',
        for j in range(num_classes):
            print str(confusion_matrix[i][j]) + '\t',
        print('')
        
    #code adopted from stackoverflow for printing confusion matrix like this
    norm_conf = []
    for i in confusion_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = confusion_matrix.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(confusion_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '0123456789'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('new_confusion_matrix.png', format='png')
    return

if __name__ == '__main__':
    flag = int(sys.argv[1])
    actual_labels = sys.argv[2]
    predicted_labels = sys.argv[3]
    predicted_test_labels = []
    actual_test_labels = []
    if(flag == 1):
        with open(actual_labels) as f:
            content = f.readlines()
            for i in range(len(content)):
                x = content[i]
                pixels = list(map(int, x.split(',')))
                num_features = len(pixels) - 1
                y = pixels[num_features]
                if not y in examples_per_class_actual.keys():
                    examples_per_class_actual[y] = [i]
                else:
                    examples_per_class_actual[y].append(i)
                actual_test_labels.append(y)
    else:
        with open(actual_labels) as f_actual:
            content = f_actual.readlines();
            for i in range(len(content)):
                y = int(content[i])
                if not y in examples_per_class_actual.keys():
                    examples_per_class_actual[y] = [i]
                else:
                    examples_per_class_actual[y].append(i)
                actual_test_labels.append(y)
    with open(predicted_labels) as f_pred:
        content = f_pred.readlines();
        for x in content:
            predicted_test_labels.append(int(x))
    num_classes = len(examples_per_class_actual.keys())
    print_confusion_matrix(examples_per_class_actual, predicted_test_labels)
