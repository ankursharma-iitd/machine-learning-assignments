from __future__ import print_function
import read_data
import time,sys,statistics,csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

class Node(object):
    def __init__(self, x_data, y_data, attribute, label, depth):
        self.x_data = x_data #feature vector in the multiway split case
        self.y_data = y_data #labels corresponding to x
        self.attribute = attribute #at what attribute (best) we splitted (if not leaf)
        self.children = [] #list of all the children nodes at which we splitted this
        self.label = label #label assigned to the node (if leaf)
        self.depth = depth #depth at each node

    def add_child(self, obj):
        self.children.append(obj)

hash_map = {} #will contain hashings of all the attributes and the number of children it can possibly split to
num_nodes = 0 #total number of nodes in the tree
train_data = np.array([])
test_data = np.array([])
valid_data = np.array([])
num_nodes_list = []
train_accuracy = []
valid_accuracy = []
test_accuracy = []
num_nodes_list_post = []
train_accuracy_post = []
valid_accuracy_post = []
test_accuracy_post = []
flag = 0
tr_x = np.array([])
tr_y = np.array([])
te_x = np.array([])
te_y = np.array([])
va_x = np.array([])
va_y = np.array([])
headnode = None
global_count = 1

def get_data_attribute(x_data, y_data, attribute, value):
    temp_x = []
    temp_y = []
    for i in range(len(y_data)):
        if(x_data[i][attribute] == value):
            temp_x.append(x_data[i])
            temp_y.append(y_data[i])
    return temp_x,temp_y

def separate_all_features(x_data, y_data, attribute):
    xhash = {}
    yhash = {}
    for i in range(len(y_data)):
        curr_attr = int(x_data[i][attribute])
        if curr_attr in xhash.keys():
            xhash[curr_attr].append(x_data[i])
            yhash[curr_attr].append(y_data[i])
        else:
            xhash[curr_attr] = [x_data[i]]
            yhash[curr_attr] = [y_data[i]]
    return xhash, yhash

def count_samples(y_data):
    countpos = 0
    for y in y_data:
        if(int(y) == 1):
            countpos += 1
    countneg = len(y_data) - countpos
    return countpos,countneg

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
        if i in x_hash.keys():
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

def get_majority_label(y_data):
    countplus, countminus = count_samples(y_data)
    if(countplus >= countminus):
        return 1
    return 0

def predict(example, head, max_depth):
    if(head.attribute == -1 or max_depth == 0):
        return head.label
    else:
        return predict(example, head.children[example[head.attribute]], max_depth - 1)

def get_accuracy(head, x_all, y_all, max_depth):
    count = 0
    for i in range(len(x_all)):
        x = x_all[i]
        y = y_all[i]
        if(y == predict(x, head, max_depth)):
            count += 1
    return ((count * 100.0) / len(x_all))

def grow_tree(x_data, y_data, available_attributes, depth, label):
    global num_nodes
    # global headnode
    global train_accuracy
    global test_accuracy
    global valid_accuracy
    global num_nodes_list
    global flag

    # print('\nAVAILABLE ATTRIBUTES : ' + ','.join([str(x) for x in available_attributes]))
    # print('CURRENT DEPTH : ' + str(depth))
    # print('DATA LENGTH : ' + str(len(x_data)))
    # print('LABEL : ' + str(label))

    sum_y = np.sum(y_data)
    if(len(y_data) == 0):
        num_nodes += 1
        # print('LEAF NODE w BLANK DATA')
        return Node(x_data, y_data, -1, label, depth)

    if(sum_y == np.shape(y_data)[0]):
        num_nodes += 1
        # print('LEAF NODE : ' + str(1))
        return Node(x_data, y_data, -1, 1, depth)

    if(sum_y == 0):
        num_nodes += 1
        # print('LEAF NODE : ' + str(0))
        return Node(x_data, y_data, -1, 0, depth)

    if(len(available_attributes) == 0):
        #net information gain on any split is 0, so stop growing the tree
        # print('LEAF NODE : ' + str(-1))
        num_nodes += 1
        return Node(x_data, y_data, -1, label, depth) #assign it the majority label

    # print('CHOOSING THE BEST ATTRIBUTE...')
    x_j,_ = choose_best_attribute(x_data, y_data, available_attributes)
    # print('BEST ATTRIBUTE CHOSEN IS : ' + str(x_j) + '\n')

    num_nodes += 1
    some_node = Node(x_data, y_data, x_j, label, depth)

    available_attributes.remove(x_j)
    x_hash,y_hash = separate_all_features(x_data, y_data, x_j)

    for i in range(hash_map[x_j]):
        if i in x_hash.keys():
            x_d = np.array(x_hash[i])
            y_d = np.array(y_hash[i])
            local_attributes = [x for x in available_attributes]
            some_node.add_child(grow_tree(x_d, y_d, local_attributes, depth + 1, get_majority_label(y_d)))
        else:
            x_d = np.array([])
            y_d = np.array([])
            local_attributes = [x for x in available_attributes]
            some_node.add_child(grow_tree(x_d, y_d, local_attributes, depth + 1, label))
    return some_node

def separate_labels(data):
    x_data = []
    y_data = []
    for somerow in data:
        y_data.append(somerow[0]) #tells you whether rich or not rich
        x_data.append(somerow[1:])
    return np.array(x_data), np.array(y_data)

def get_num_nodes(head, dictionary):
    curr_depth = head.depth
    if curr_depth in dictionary.keys():
        dictionary[curr_depth] += 1
    else:
        dictionary[curr_depth] = 1
    if(len(head.children) > 0):
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

def get_num_of_nodes(node, count):
    if(node.attribute == -1):
        return 1
    for child in node.children:
        count += get_num_of_nodes(child, 0)
    return (count + 1)

def plot_pruned_graphs():
    global global_count
    global num_nodes_list_post
    global train_accuracy_post
    global test_accuracy_post
    global valid_accuracy_post
    #plotting the curves
    print('\nPLOTTING GRAPH WHILE PRUNING THE TREE...')
    fig = plt.figure()
    plt.plot(num_nodes_list_post, train_accuracy_post, color='b', label='Training Accuracy')
    plt.plot(num_nodes_list_post, valid_accuracy_post, color='r', label='Validation Accuracy')
    plt.plot(num_nodes_list_post, test_accuracy_post, color='g', label='Testing Accuracy')
    plt.legend()
    plt.xlabel('Number of Nodes')
    plt.ylabel('Accuracy (in %)')
    plt.title('Variation of Accuracy with the Number of nodes (after post-pruning)')
    fig.savefig('graphs/pruned_graph_' + str(global_count), dpi=fig.dpi)
    plt.close(fig)
    print('GRAPH SAVED!\n')

def post_order_traversal(node):
    global headnode
    global flag
    global num_nodes
    global num_nodes_list_post
    global train_accuracy_post
    global test_accuracy_post
    global valid_accuracy_post
    global global_count
    if(flag == 0):
        headnode = node
        flag = 1

    # if(node.attribute == -1): #leaf node
    #     return node

    for i in range(len(node.children)):
        node.children[i] = post_order_traversal(node.children[i])

    old_accuracy = get_accuracy(headnode, va_x, va_y, sys.maxint)
    # print('\nLEAF/NON-LEAF :' + str(node.attribute))
    # print('OLD ACCURACY : ' + str(old_accuracy))

    temp = copy.deepcopy(node) #store a temp copy of the node
    #replace the entire subtree rooted at node with a leaf node
    node.children = None
    node.attribute = -1
    temp_nodes = num_nodes
    pruned_num = get_num_of_nodes(temp, 0)
    num_nodes = temp_nodes - pruned_num + 1

    new_accuracy = get_accuracy(headnode, va_x, va_y, sys.maxint)
    # print('NEW ACCURACY : ' + str(new_accuracy))
    improvement = new_accuracy - old_accuracy
    # print('TO BE PRUNED NUMBER OF NODES : ' + str(pruned_num))
    if(improvement >= 0):
        print(global_count)
        train_accuracy_post.append(get_accuracy(headnode, tr_x, tr_y, sys.maxint))
        test_accuracy_post.append(get_accuracy(headnode, te_x, te_y, sys.maxint))
        valid_accuracy_post.append(new_accuracy)
        num_nodes_list_post.append(num_nodes)
        if((global_count % 100) == 0):
            plot_pruned_graphs()
        global_count += 1
    else: #check for improvement, if no improvement then undo the pruning else prune it
        node = temp #undo the pruning
        num_nodes = temp_nodes
    # print('FINAL NUMBER OF NODES :' + str(num_nodes) + '\n')
    return node

def remove_the_data(tree):
    tree.x_data = None
    tree.y_data = None
    for i in range(len(tree.children)):
        remove_the_data(tree.children[i])
    return

if __name__ == '__main__':
    
    train_data,valid_data,test_data,hash_thing = read_data.init()
    hash_map = hash_thing
    print(hash_map)
    tr_x, tr_y = separate_labels(train_data)
    te_x, te_y = separate_labels(test_data)
    va_x, va_y = separate_labels(valid_data)
    initial_attrs = hash_map.keys()

    # f = open('train_x.csv', 'w+')
    # for i in range(len(x_data)):
    #     curr_data = x_data[i]
    #     somestring = ''
    #     for j in range(len(curr_data)):
    #         somestring += (str(j) + ':' + str(curr_data[j]) + ',')
    #     f.write(somestring + '\n')
    # f.close()

    # f = open('train_y.csv', 'w+')
    # for i in range(len(y_data)):
    #     somestring = str(y_data[i])
    #     f.write(somestring + '\n')
    # f.close()

    print('\nGROWING TREE...')
    head = grow_tree(np.array(tr_x), np.array(tr_y), initial_attrs, 1, get_majority_label(tr_y))
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

    print('\nNUMBER OF NODES IN THE FULLY GROWN TREE : ' + str(num_nodes) + '\n')
    # print('\nNUMBER OF NODES IN THE FULLY GROWN TREE : ' + str(get_num_of_nodes(head, 0)) + '\n')

    # post_pruning the nodes
    print('\nPRUNING THE NODES...')
    newtree = copy.deepcopy(head)
    remove_the_data(newtree)
    prunedhead = post_order_traversal(newtree)
    print('PRUNING DONE!\n')

    print('PLOTTING THE PRUNED GRAPHS...')
    plot_pruned_graphs()
    print('PLOTTING DONE!')

    print('\nAFTER PRUNING...')
    print('NUMBER OF NODES IN THE TREE : ' + str(num_nodes))
    print('TRAINING SET ACCURACY : ' + str(get_accuracy(prunedhead, tr_x, tr_y, sys.maxint)))
    print('TESTING SET ACCURACY : ' + str(get_accuracy(prunedhead, te_x, te_y, sys.maxint)))
    print('VALIDATION SET ACCURACY : ' + str(get_accuracy(prunedhead, va_x, va_y, sys.maxint))+ '\n')
