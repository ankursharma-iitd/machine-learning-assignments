from __future__ import print_function
import time, sys, statistics, csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

## The possible attributes in the data with the prediction at index 0. Smaller names for brevity.
attributes = ["rich","age","wc","fnlwgt","edu","edun","mar","occ","rel","race","sex","capg","canpl","hpw","nc"]

## Get the encoding of the csv file by replacing each categorical attribute value by its index.
wc_l = "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked".split(", ")
edu_l = "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool".split(", ")
mar_l = "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse".split(", ")
occ_l = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces".split(", ")
rel_l = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried".split(", ")
race_l = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black".split(", ")
sex_l = "Female, Male".split(", ")
nc_l = "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split(", ")
encode = {
    "rich"   : {"0":0,"1":1},
    "wc"     : {wc_l[i]:i for i in range(len(wc_l))},
    "edu"    : {edu_l[i]:i for i in range(len(edu_l))},
    "mar"    : {mar_l[i]:i for i in range(len(mar_l))},
    "occ"    : {occ_l[i]:i for i in range(len(occ_l))},
    "rel"    : {rel_l[i]:i for i in range(len(rel_l))},
    "race"   : {race_l[i]:i for i in range(len(race_l))},
    "sex"    : {sex_l[i]:i for i in range(len(sex_l))},
    "nc"     : {nc_l[i]:i for i in range(len(nc_l))},
    }

#defining some globals
hash_map = {0:2, 1:len(wc_l), 2:2 , 3:len(edu_l), 4:2, 5:len(mar_l), 6:len(occ_l), 7:len(rel_l), 8:len(race_l), 9:len(sex_l), 10:2, 11:2, 12:2, 13:len(nc_l)}
train_data = np.array([])
test_data = np.array([])
valid_data = np.array([])
tr_x = np.array([])
tr_y = np.array([])
te_x = np.array([])
te_y = np.array([])
va_x = np.array([])
va_y = np.array([])
headnode = None
num_nodes_list = []
train_accuracy = []
valid_accuracy = []
test_accuracy = []
num_nodes = 0 #total number of nodes in the tree
flag = 0
global_count = 0
numerical_attributes = []
discrete_attributes = []
reporting_max_hash = {}
reporting_thresh_hash = {}

class Node(object):
    def __init__(self, x_data, y_data, attribute, label, depth, median):
        self.x_data = x_data  #feature vector in the multiway split case
        self.y_data = y_data  #labels corresponding to x
        self.attribute = attribute  #at what attribute (best) we splitted (if not leaf)
        self.children = []  #list of all the children nodes at which we splitted this
        self.label = label  #label assigned to the node (if leaf)
        self.depth = depth  #depth at each node
        self.median = median #median where the split has taken place

    def add_child(self, obj):
        self.children.append(obj)

def preprocess(file):
    """
    Given a file, read its data by encoding categorical attributes and binarising continuos attributes based on median.
    params(1): 
        file : string : the name of the file
    outputs(6):
        2D numpy array with the data
    """
    fin = open(file, "r")
    reader = csv.reader(fin)
    data = []
    total = 0
    for row in reader:
        total += 1
        # Skip line 0 in the file
        if (total == 1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        t = [0 for i in range(15)]

        # Encode the categorical attributes
        t[0] = encode["rich"][l[-1]]
        t[2] = encode["wc"][l[1]]
        t[4] = encode["edu"][l[3]]
        t[6] = encode["mar"][l[5]]
        t[7] = encode["occ"][l[6]]
        t[8] = encode["rel"][l[7]]
        t[9] = encode["race"][l[8]]
        t[10] = encode["sex"][l[9]]
        t[14] = encode["nc"][l[13]]

        # Modify this section to read the file in part c where you split the continuos attributes baed on dynamic median values.
        t[1] = float(l[0])
        t[3] = float(l[2])
        t[5] = float(l[4])
        t[11] = float(l[10])
        t[12] = float(l[11])
        t[13] = float(l[12])

        # Convert some of the booleans to ints
        data.append([int(x) for x in t])

    return np.array(data, dtype=np.int64)

def init():
    print('\r\nPREPROCESSING STARTED...')
    ## Read the data
    train_data = preprocess("dtree_data/train.csv")
    valid_data = preprocess("dtree_data/valid.csv")
    test_data = preprocess("dtree_data/test.csv")
    print("The sizes are ","Train:",len(train_data),", Validation:",len(valid_data),", Test:",len(test_data))
    print('PREPROCESSING DONE!\r\n')
    return train_data,valid_data,test_data,hash_map

def separate_labels(data):
    x_data = []
    y_data = []
    for somerow in data:
        y_data.append(somerow[0])  #tells you whether rich or not rich
        x_data.append(somerow[1:])
    return np.array(x_data), np.array(y_data)

def count_samples(y_data):
    countpos = 0
    for y in y_data:
        if(int(y) == 1):
            countpos += 1
    countneg = len(y_data) - countpos
    return countpos,countneg

def get_majority_label(y_data):
    countplus, countminus = count_samples(y_data)
    if (countplus >= countminus):
        return 1
    return 0

def separate_all_features(x_data, y_data, attribute):
    xhash = {}
    yhash = {}

    #empty initialisation
    for i in range(hash_map[attribute]):
        xhash[i] = []
        yhash[i] = []

    binlist = []
    if attribute in numerical_attributes:
        binlist = get_median(x_data[:, attribute])

    for i in range(len(x_data)):
        if attribute in discrete_attributes:
            idx = int(x_data[i][attribute])
        else:
            idx = int(binlist[i])
        xhash[idx].append(x_data[i])
        yhash[idx].append(y_data[i])
    return xhash, yhash

def get_median(somelist):
    newlist = []
    median = statistics.median(somelist)
    for elem in somelist:
        if(elem >= median):
            newlist.append(1)
        else:
            newlist.append(0)
    return newlist

def get_entropy(x_data, y_data):
    countpos,countneg = count_samples(y_data)
    py_0 = (countneg * 1.0) / (countneg + countpos)
    py_1 = 1.0 - py_0
    try:
        val = ((py_0 * math.log(py_0, 2)) + (py_1 * math.log(py_1, 2)))
    except:
        val = 0.0 #takes the value 0 in the limiting case
    return (-1.0 * val)

def get_H_given_X_j(x_data, y_data, attribute):
    x_hash,y_hash = separate_all_features(x_data, y_data, attribute)
    sum = 0.0
    for i in range(hash_map[attribute]):
        if(len(x_hash[i]) > 0):
            sum += (((len(y_hash[i]) * 1.0)/len(y_data)) * (get_entropy(x_hash[i], y_hash[i])))
    return sum

def get_information_measure(x_data, y_data, attribute):
    return (get_entropy(x_data, y_data) - get_H_given_X_j(x_data, y_data, attribute))

def choose_best_attribute(x_data, y_data, all_available_attributes):
    chosen_attr = all_available_attributes[0] #default assign
    chosen_info = -sys.maxint - 1
    sum_info = 0.0
    #write the logic to select the best attribut at a given node among the list of available attributes
    for attribute in all_available_attributes:
        info = get_information_measure(x_data, y_data, attribute)
        sum_info += info
        if(info > chosen_info):
            chosen_info = info
            chosen_attr = attribute
    return chosen_attr,sum_info

def grow_tree(x_data, y_data, available_attributes, depth, label):
    global num_nodes
    # global headnode
    global train_accuracy
    global test_accuracy
    global valid_accuracy
    global num_nodes_list
    global flag

    print('\nAVAILABLE ATTRIBUTES : ' + ','.join([str(x) for x in available_attributes]))
    print('CURRENT DEPTH : ' + str(depth))
    print('DATA LENGTH : ' + str(len(x_data)))
    print('LABEL : ' + str(label))

    sum_y = np.sum(y_data)
    if(len(y_data) == 0):
        num_nodes += 1
        print('LEAF NODE w BLANK DATA')
        return Node(x_data, y_data, -1, label, depth, -1)

    if(sum_y == np.shape(y_data)[0]):
        num_nodes += 1
        print('LEAF NODE : ' + str(1))
        return Node(x_data, y_data, -1, 1, depth, -1)

    if(sum_y == 0):
        num_nodes += 1
        print('LEAF NODE : ' + str(0))
        return Node(x_data, y_data, -1, 0, depth, -1)

    if(len(available_attributes) == 0):
        #net information gain on any split is 0, so stop growing the tree
        print('LEAF NODE : ' + str(-1))
        num_nodes += 1
        return Node(x_data, y_data, -1, label, depth, -1) #assign it the majority label

    print('CHOOSING THE BEST ATTRIBUTE...')
    x_j,_ = choose_best_attribute(x_data, y_data, available_attributes)
    print('BEST ATTRIBUTE CHOSEN IS : ' + str(x_j) + '\n')

    #taking care of the special case
    if(x_j in numerical_attributes):
        median = statistics.median(x_data[:, x_j])
    else:
        median = -1

    if ((x_j in numerical_attributes) and ((len(y_data) >= 2) and (np.sum(get_median(x_data[:, x_j])) == 0 or np.sum(get_median(x_data[:, x_j])) == len(x_data)) )):
        num_nodes += 1
        print('LEAF NODE w SPECIAL CASE')
        return Node(x_data, y_data, -1, get_majority_label(y_data), depth, -1)

    num_nodes += 1
    some_node = Node(x_data, y_data, x_j, label, depth, median)

    #if it is discrete it cannot be repeated more than once along the path from root to leaf
    if((x_j in discrete_attributes) or (len(x_data) == 1)):
        available_attributes.remove(x_j)

    x_hash,y_hash = separate_all_features(x_data, y_data, x_j)

    for i in range(hash_map[x_j]):
        local_attributes = copy.deepcopy(available_attributes)
        x_d = np.array(x_hash[i])
        y_d = np.array(y_hash[i])
        if(len(y_d) == 0):
            some_node.add_child(grow_tree(x_d, y_d, local_attributes, depth + 1, label))
        else:
            some_node.add_child(grow_tree(x_d, y_d, local_attributes, depth + 1, get_majority_label(y_d)))
    return some_node

def predict(example, head, max_depth):
    if(head.attribute == -1 or max_depth == 0):
        return head.label
    else:
        if(head.attribute in discrete_attributes):
            return predict(example, head.children[example[head.attribute]], max_depth - 1)
        else:
            pos = int(example[head.attribute] >= head.median)
            return predict(example, head.children[pos], max_depth - 1)

def get_accuracy(head, x_all, y_all, max_depth):
    count = 0
    for i in range(len(x_all)):
        x = x_all[i]
        y = y_all[i]
        if(y == predict(x, head, max_depth)):
            count += 1
    return ((count * 100.0) / len(x_all))


def get_num_nodes(head, dictionary):
    curr_depth = head.depth
    if curr_depth in dictionary.keys():
        dictionary[curr_depth] += 1
    else:
        dictionary[curr_depth] = 1
    if (len(head.children) > 0):
        for child in head.children:
            dictionary = get_num_nodes(child, dictionary)
    return dictionary


def plot_graphs(head):
    global num_nodes_list
    global train_accuracy
    global test_accuracy
    global valid_accuracy
    global tr_x
    global tr_y
    global te_x
    global te_y
    global va_x
    global va_y
    sum_nodes = 0
    node_hash = get_num_nodes(head, {})
    print(node_hash)
    for i in node_hash.keys():
        depth = i
        sum_nodes += node_hash[i]
        train_accuracy.append(get_accuracy(head, tr_x, tr_y, depth))
        test_accuracy.append(get_accuracy(head, te_x, te_y, depth))
        valid_accuracy.append(get_accuracy(head, va_x, va_y, depth))
        num_nodes_list.append(sum_nodes)

    plt.figure(1)
    plt.plot(num_nodes_list, train_accuracy, color='b', label='Training Accuracy')
    plt.plot(num_nodes_list, valid_accuracy, color='r', label='Validation Accuracy')
    plt.plot(num_nodes_list, test_accuracy, color='g', label='Testing Accuracy')
    plt.legend()
    plt.xlabel('Number of Nodes')
    plt.ylabel('Accuracy (in %)')
    plt.title('Variation of Accuracy with the Number of nodes')
    plt.show()
    return

def report_max_findings(node, attr):
    if(node.attribute == attr):
        val = 1
    else:
        val = 0
    max = 0
    for child in node.children:
        something = report_max_findings(child, attr)
        if(something > max):
            max = something
    return (max + val)

def report_threshold_findings(node, attr, thresholds_list):
    if(node.attribute == attr):
        flag = 1
    else:
        flag = 0
    maxlist = []
    for child in node.children:
        somelist = report_threshold_findings(child, attr, [])
        if(len(somelist) > len(maxlist)):
            maxlist = somelist
    if(flag == 1):
        maxlist.append(node.median)
    return maxlist

if __name__ == '__main__':
    train_data, valid_data, test_data, hash_map =init()
    tr_x, tr_y = separate_labels(train_data)
    te_x, te_y = separate_labels(test_data)
    va_x, va_y = separate_labels(valid_data)
    initial_attrs = hash_map.keys()
    numerical_attributes = [0, 2, 4, 10, 11, 12]
    discrete_attributes = [1, 3, 5, 6, 7, 8, 9, 13]

    # f = open('train_x_part_c.csv', 'w+')
    # for i in range(len(tr_x)):
    #     curr_data = tr_x[i]
    #     somestring = ''
    #     for j in range(len(curr_data)):
    #         somestring += (str(j) + ':' + str(curr_data[j]) + ',')
    #     f.write(somestring + '\n')
    # f.close()

    # print(tr_x[:, 0])
    # tr_x[:, 0] = get_median(tr_x[:, 0])
    # print(tr_x[:, 0])

    print('\nGROWING TREE...')
    sys.setrecursionlimit(50000)
    head = grow_tree(tr_x, tr_y, initial_attrs, 1, get_majority_label(tr_y))
    print('DONE!\n')

    print('\nBEFORE PRUNING...')
    print('NUMBER OF NODES IN THE TREE : ' + str(num_nodes))
    print('TRAINING SET ACCURACY : ' + str(get_accuracy(head, tr_x, tr_y, sys.maxint)))
    print('TESTING SET ACCURACY : ' + str(get_accuracy(head, te_x, te_y, sys.maxint)))
    print('VALIDATION SET ACCURACY : ' + str(get_accuracy(head, va_x, va_y, sys.maxint))+ '\n')

    #plotting the curves
    print('\nPLOTTING GRAPH...')
    plot_graphs(head)
    print('GRAPH PLOTTED!\n')

    #Report the thresholds and what numerical attributes split has taken place
    print('\nREPORTING ITEMS...')
    for attribute in numerical_attributes:
        reporting_max_hash[attribute] = report_max_findings(head, attribute)
        reporting_thresh_hash[attribute] = report_threshold_findings(head, attribute, [])
    print(reporting_max_hash)
    print(reporting_thresh_hash)
    print('REPORTED!\n')
