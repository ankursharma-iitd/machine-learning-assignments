import sys
import pickle
from collections import Counter
import operator
from svmutil import *
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import string
import math

model_number = int(sys.argv[1])
input_name = sys.argv[2]
output_name = sys.argv[3]
corresponding_model = 'models_1/model' + sys.argv[1] + '.p'
test_data = []
num_examples = 0
parameters_y = {} #hashtable for the aprior probabilities for each class
parameters_x_given_y = {} #integer hash table that stores the parameters for each class and for each word in the vocab i.e params[class][word]
downsums = {} #store the downsum values of each class as a hashtable
all_classes = [] #list of all the classses

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
            stopped_tokens = [
                token for token in tokens if token not in en_stop
            ]
            stemmed_tokens = [
                p_stemmer.stem(token) for token in stopped_tokens
            ]
            documentWords = ' '.join(stemmed_tokens)
            print((documentWords), file=out)
        out.close()
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
    y = []  #initialising the prediction array
    for each_x in data:
        predicted_class = '0'
        max_probab = float("-inf")
        for each_class in all_classes:
            curr_prob = get_probability(each_x, each_class)
            if (curr_prob > max_probab):
                predicted_class = each_class
                max_probab = curr_prob
        y.append(predicted_class)
    return y


def get_accuracy(prediction, actual):
    length = len(prediction)
    sum = 0.0
    for i in range(length):
        if (int(prediction[i]) == int(actual[i])):
            sum += 1.0
    return ((sum * 100.0) / length)


def remove_numbers(str1):
    reg_compile = re.compile('[0-9]*')
    return re.sub(reg_compile, '', str1)


def remove_tags(str2):
    reg_compile2 = re.compile('<.*?>')
    newtext = re.sub(reg_compile2, '', str2)
    return newtext

#method that can be used for feature engineering for k-gram features
def kgram_feature_vector(some_string, k):
    word_list = re.findall(r'\w+', some_string)
    if (k == 1):
        return word_list
    new_list = []
    for i in range(len(word_list) - k + 1):
        some_string = ''
        for j in range(k):
            some_string += word_list[i + j] + ' '
        new_list.append(some_string.strip())
    return new_list


def get_hash_with_min_count(somecounter, mincount):
    if (min_count == 0):
        return somecounter
    keys = somecounter.keys()
    newhash = {}
    for key in keys:
        if (somecounter[key] >= mincount):
            newhash[key] = somecounter[key]
    return newhash


def filter_with_length(allwords, min_length):
    if (min_length == 1):
        return allwords
    words = []
    for word in allwords:
        if (len(word) >= min_length):
            words.append(word)
    return words


if __name__ == '__main__':
    if (model_number == 1 or model_number == 2 or model_number == 3):

        #Default initialisation of some parameters
        flag = 0  #make this flag 1 to remove numbers
        min_count = 1  #set the min count of the word to be considered in the feature vector
        k_gram = 1  #set how many words to take together
        min_length = 1  #set the min length of each word
        stems_bigrams_of_stems = 0  #add bigrams and stems to get new features
        #use the bigram features with stop word removal in Model 3
        if (model_number == 3):
            k_gram = 2

        # test_labels = 'imdb/imdb_test_labels.txt'
        # test_label = []
        # print('\nREADING THE TEST LABEL FILE...')
        # with open(test_labels) as f:
        #     content = f.readlines()
        #     for i in range(len(content)):
        #         y = content[i]
        #         test_label.append(y)
        # print('TEST LABEL FILE READ SUCCESSFULLY!\n')

        #get remove stopwords and stemming in Model 2, and Model 1
        if (model_number != 1):
            test_text = 'new_test_text'
            getStemmedDocument(input_name, test_text)
            print('\nSTEMMING DONE AND STOPWORDS REMOVED!')
        else:
            test_text = input_name

        print('\nREADING THE TEST DATA FILE...')
        flag = 0
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

        num_examples = len(test_data)
        file = open(corresponding_model, 'rb')
        parameters_y = pickle.load(file)
        parameters_x_given_y = pickle.load(file)
        downsums = pickle.load(file)
        all_classes = pickle.load(file)

        print('PREDICTION..')
        #doi the prediction here
        test_prediction = predict(test_data) #get the prediction for test data
        # test_accuracay = get_accuracy(test_prediction, test_label)
        # print(test_accuracay)

        with open(output_name, 'w+') as f:
            for each_prediction in test_prediction:
                string = str(each_prediction) + '\n'
                f.write(string)
            f.close()
    else:
        print('INCORRECT MODEL NUMBER. EXITING...')
        exit(0)