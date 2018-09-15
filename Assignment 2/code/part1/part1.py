import re
import os
import string
import math
import sys
from collections import Counter
import random
import time
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

num_examples = 0 #total number of examples in training set
examples_per_class = {} #store the examples relevant to each class as a hashtbale
examples_per_class_test = {} #store relevant examples for test per class
vocab = set() #vocabulary containing all the words
bags = [] #contains the list of bag of words for each document
num_classes = 0 #total number of classes
parameters_y = {} #hashtable for the aprior probabilities for each class
parameters_x_given_y = {} #integer hash table that stores the parameters for each class and for each word in the vocab i.e params[class][word]
downsums = {} #store the downsum values of each class as a hashtable
train_data = [] #training data set
train_label = [] #labels for the training set
test_data = [] #array that stores the 2D array containing the test documents
test_label = [] #array containing the actual results of the test data
confusion_matrix = [[]] #confusion matrix is a 2D array

#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

# function that takes an input file and performs stemming to generate the output file
def getStemmedDocument(inputFileName, outputFileName):
    out = open(outputFileName, 'w')
    with open(inputFileName) as f:
        docs = f.readlines()
        for doc in docs:
            raw = doc.lower()
            raw = raw.replace("<br /><br />", " ")
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [token for token in tokens if token not in en_stop]
            stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
            documentWords = ' '.join(stemmed_tokens)
            print((documentWords), file=out)
        out.close()
    return

#an instance of this class if the bag of word for each text document
class BagOfWords:
    def __init__(self):
        self.frequency_table = {} #maintains word count for each word in a given text document
        self.num_of_words = 0

    def get_table(self):
        return self.frequency_table

    def get_num_of_words(self):
        return self.num_of_words

#this will enable you to learn the parameters
#assuming the same multinomial distribution each word
def learn():
    # print('LEARNING STARTED : ')
    for each_class in examples_per_class.keys():
        parameters_y[each_class] = (len(examples_per_class[each_class]) * 1.0)/num_examples

    c = 1.0
    for each_class in examples_per_class.keys():
        downsum = len(vocab) * c
        upsum = {}
        for doc in examples_per_class[each_class]:
            word_table = bags[doc].frequency_table
            for word in word_table.keys():
                # print('WORD : ' + word + ', CLASS : ' + each_class)
                if not word in upsum:
                    upsum[word] = c
                upsum[word] += word_table[word]
            downsum += bags[doc].num_of_words
        for word in upsum.keys():
            upsum[word] /= downsum
        parameters_x_given_y[each_class] = upsum
        downsums[each_class] = downsum
    return

def get_probability(test_x, given_class):
    aprior_prob = parameters_y[given_class]
    downsum = downsums[given_class]
    x_given_y = parameters_x_given_y[given_class]
    prod = math.log(aprior_prob * 1.0)
    for word in test_x:
        if word in x_given_y:
            prod += math.log(x_given_y[word])
        else:
            prod -= math.log(downsum)
    return prod

def predict(data):
    y = [] #initialising the prediction array
    for each_x in data:
        predicted_class = '0'
        max_probab = float("-inf")
        for each_class in examples_per_class.keys():
            curr_prob = get_probability(each_x, each_class)
            if(curr_prob > max_probab):
                predicted_class = each_class
                max_probab = curr_prob
        y.append(predicted_class)
    return y

def get_accuracy(prediction, actual):
    length = len(prediction)
    sum = 0.0
    for i in range(length):
        if(prediction[i] == actual[i]):
            sum += 1.0
    return ((sum * 100.0) / length)

def remove_numbers(str1):
    reg_compile = re.compile('[0-9]*')
    return re.sub(reg_compile, '', str1)

def remove_tags(str2):
    reg_compile2 = re.compile('<.*?>')
    newtext = re.sub(reg_compile2, '', str2)
    return newtext

def get_random_accuracy(classes):
    arr = []
    classes = list(classes)
    for i in range(num_examples):
        randomclass = classes[random.randint(0,7)]
        arr.append(randomclass)
    return arr

def get_majority_class(labels):
    words = Counter(labels)
    maj_class = '0'
    count = float("-inf")
    for each_class in words.keys():
        if(words[each_class] > count):
            count = words[each_class]
            maj_class = each_class
    return maj_class

def print_confusion_matrix(predicted_test_labels, examples_per_class_test):
    classes = list(examples_per_class_test.keys())
    confusion_matrix = [[0 for i in range(num_classes)] for j in range(num_classes)]
    for i in range(num_classes):
        each_class = classes[i]
        corresponding_examples = examples_per_class_test[each_class]
        for each_example in corresponding_examples:
            predicted_class = predicted_test_labels[each_example]
            j = classes.index(predicted_class)
            confusion_matrix[j][i] += 1

    print('\t\t\t\t\tA  C  T  U  A  L   \t\tC  L  A  S  S  E  S')
    print('\t\t'+'\t\t '.join(classes))
    for i in range(num_classes):
        print(classes[i] + '\t\t', end = '')
        for j in range(num_classes):
            print(str(confusion_matrix[i][j]) + '\t\t', end = '')
        print('')

    return

#method that can be used for feature engineering for k-gram features
def kgram_feature_vector(some_string, k):
    word_list = re.findall(r'\w+', some_string)
    if(k == 1):
        return word_list
    new_list = []
    for i in range(len(word_list) - k + 1):
        some_string = ''
        for j in range(k):
            some_string += word_list[i + j] + ' '
        new_list.append(some_string.strip())
    return new_list

def get_hash_with_min_count(somecounter, mincount):
    if(min_count == 0):
        return somecounter
    keys = somecounter.keys()
    newhash = {}
    for key in keys:
        if(somecounter[key] >= mincount):
            newhash[key] = somecounter[key]
    return newhash

def filter_with_length(allwords, min_length):
    if(min_length == 1):
        return allwords
    words = []
    for word in allwords:
        if(len(word) >= min_length):
            words.append(word)
    return words

# def termfreq(matrix, doc, term):
#     try:
#         return matrix[doc][term] / float(sum(matrix[doc].values()))
#     except ZeroDivisionError: return 0
# def inversedocfreq(matrix, term):
#     try:
#         return float(len(matrix)) /sum([1 for i,_ in enumerate(matrix) if matrix[i][term] > 0])
#     except ZeroDivisionError: return 0

if __name__ == '__main__':
    #data reading and cleaning work

    model_to_use = int(sys.argv[1])
    train_text = sys.argv[2]
    train_labels = sys.argv[3]
    test_text = sys.argv[4]
    test_labels = sys.argv[5]

    #Default initialisation of some parameters
    flag = 0 #make this flag 1 to remove numbers
    min_count = 1 #set the min count of the word to be considered in the feature vector
    k_gram = 1 #set how many words to take together
    min_length = 1 #set the min length of each word
    stems_bigrams_of_stems = 0 #add bigrams and stems to get new features
    #use the bigram features with stop word removal in Model 3
    if(model_to_use == 3):
        k_gram = 2

    #get remove stopwords and stemming in Model 2, and Model 1
    if(model_to_use != 1):
        train_text = 'new_train_text'
        # getStemmedDocument(sys.argv[2], train_text)
        test_text = 'new_test_text'
        # getStemmedDocument(sys.argv[4], test_text)
        print('\nSTEMMING DONE AND STOPWORDS REMOVED!')


    print('\nREADING THE TRAIN DATA FILE...')
    with open(train_text) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        for x in content:
            if flag == 1:
                x = remove_numbers(x)
            cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', remove_tags(x).lower())
            words = kgram_feature_vector(cleaned_text, k_gram)
            if(stems_bigrams_of_stems == 1):
                words = words + kgram_feature_vector(
                    cleaned_text, 1) + kgram_feature_vector(cleaned_text, 3)
            words = filter_with_length(words, min_length)

            # text = x.strip().lower().translate(table, string.punctuation)
            bag = BagOfWords() #create the bow for this text
            bag.num_of_words = len(words)
            word_count = {}
            for word in words:
                vocab.add(word)
            bag.frequency_table = get_hash_with_min_count(Counter(words), min_count)
            bags.append(bag) #add this bag to the list of all bags
            train_data.append(words)
    print('TRAIN DATA FILE READ SUCCESSFULLY!\n')

    print('\nREADING THE TRAIN LABEL FILE...')
    with open(train_labels) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        for i in range(len(content)):
            y = content[i].strip()
            if not y in examples_per_class.keys():
                examples_per_class[y] = [i]
            else:
                examples_per_class[y].append(i);
            num_examples += 1
            train_label.append(y)
    print('TRAIN LABEL FILE READ SUCCESSFULLY!\n')
    num_classes = len(examples_per_class.keys())
    #at this point your all the data structures are intialised with the values i.e. one time text processing has been done

    print('\nLEARNING THE PARAMETERS OF NAIVE BAYES MODEL...')
    time_start = time.clock()
    learn() #this will make the model learn the parameters based on this training text
    time_learning = time.clock() - time_start
    print('PARAMETERS LEARNT!\n')

    print('\nREADING THE TEST DATA FILE...')
    with open(test_text) as f:
        content = f.readlines()
        for x in content:
            if flag == 1:
                x = remove_numbers(x)
            cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', remove_tags(x.lower()))
            words = kgram_feature_vector(cleaned_text, k_gram)
            if(stems_bigrams_of_stems == 1):
                words = words + kgram_feature_vector(
                    cleaned_text, 1) + kgram_feature_vector(cleaned_text, 3)
            words = filter_with_length(words, min_length)
            test_data.append(words)
    print('TEST DATA FILE READ SUCCESSFULLY!\n')

    print('\nREADING THE TEST LABEL FILE...')
    with open(test_labels) as f:
        content = f.readlines()
        for i in range(len(content)):
            y = content[i].strip()
            if not y in examples_per_class_test.keys():
                examples_per_class_test[y] = [i]
            else:
                examples_per_class_test[y].append(i);
            test_label.append(y)
    print('TEST LABEL FILE READ SUCCESSFULLY!\n')

    print('\nTRAINING SET PREDICTION...')
    time_start = time.clock()
    prediction_train = predict(train_data) #get the prediction for training data using the parameters computed earlier
    time_train_prediction = time.clock() - time_start

    print('\nTEST SET PREDICTION...')
    time_start = time.clock()
    prediction_test = predict(test_data) #get the prediction for test data
    time_test_prediction = time.clock() - time_start

    #compute accuracy for the Naive Bayes Model
    train_accuracy = get_accuracy(prediction_train, train_label)
    test_accuracy = get_accuracy(prediction_test, test_label)

    print('\nACCURACIES OBTAINED FROM NAIVA-BAYES MODEL : ')
    print('TRAINING SET ACCURACY : ' + str(train_accuracy))
    print('TEST SET ACCURACY : ' + str(test_accuracy))

    #compute accuracies by random prediction
    random_labels = get_random_accuracy(examples_per_class.keys())
    print('\nACCURACIES OBTAINED FROM RANDOM PREDICTION : ')
    print('TRAINING SET ACCURACY : ' +
          str(get_accuracy(random_labels, train_label)))
    print('TEST SET ACCURACY : ' +
          str(get_accuracy(random_labels, test_label)))

    #compute accuracies by majority prediction
    majority_class_train = get_majority_class(train_label)
    majority_class_test = get_majority_class(test_label)
    majority_labels_train = [majority_class_train for i in range(num_examples)]
    majority_labels_test = [majority_class_test for i in range(num_examples)]
    print('\nACCURACIES OBTAINED FROM MAJORITY PREDICTION : ')
    print('TRAINING SET ACCURACY : ' +
          str(get_accuracy(majority_labels_train, train_label)))
    print(
        'TEST SET ACCURACY : ' + str(get_accuracy(majority_labels_test, test_label)))

    #this will print the confusion matrix on the screen
    print('\n\nCONFUSION MATRIX FOR THE TRAIN DATA: \n')
    print_confusion_matrix(prediction_train, examples_per_class)
    print('\n\nCONFUSION MATRIX FOR THE TEST DATA: \n')
    print_confusion_matrix(prediction_test, examples_per_class_test)

    #this will print the timing details
    print('\n\nTIMING DETAILS :\n')
    print('LEARN THE PARAMETERES : '+ str(time_learning))
    print('TRAINING SET PREDICTION : ' + str(time_train_prediction))
    print('TEST SET PREDICTION : ' + str(time_test_prediction))


    #save the model onto a file
    file = open('new_model' + str(model_to_use) + '.p', 'wb')
    pickle.dump(parameters_y, file)
    pickle.dump(parameters_x_given_y, file)
    pickle.dump(downsums, file)
    all_classes = list(examples_per_class.keys())
    pickle.dump(all_classes, file)
    file.close()



    # #feature engineering last attempt
    # tf_idf = [[0.0 for i in range(len(vocab))] for j in range(num_examples)]
    # matrix = []
    # for i in range(num_examples):
    #     print('RUNNING FOR i : ' + str(i))
    #     curr = {}
    #     somehash = bags[i].frequency_table
    #     for word in vocab:
    #         val = 0
    #         if word in somehash.keys():
    #             val = somehash[word]
    #         curr[word] = val
    #     matrix.append(curr)
    # print(matrix[0])
    # print(matrix[1])
