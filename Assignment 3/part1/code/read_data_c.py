"""
This file gives you the code to read the data into numpy arrays to get you startedf for part (a).
"""
from __future__ import print_function
import time,sys,statistics,csv
import numpy as np

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
hash_map = {0:2, 1:len(wc_l), 2:2 , 3:len(edu_l), 4:2, 5:len(mar_l), 6:len(occ_l), 7:len(rel_l), 8:len(race_l), 9:len(sex_l), 10:2, 11:2, 12:2, 13:len(nc_l)}

def preprocess(file):
    """
    Given a file, read its data by encoding categorical attributes and binarising continuos attributes based on median.
    params(1): 
        file : string : the name of the file
    outputs(6):
        2D numpy array with the data
    """
    fin = open(file,"r")
    reader = csv.reader(fin)
    data = []
    total = 0
    for row in reader:
        total+=1
        # Skip line 0 in the file
        if(total==1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        t = [0 for i in range(15)]

        # Encode the categorical attributes
        t[0] = encode["rich"][l[-1]]; t[2] = encode["wc"][l[1]]; t[4] = encode["edu"][l[3]]
        t[6] = encode["mar"][l[5]]; t[7] = encode["occ"][l[6]]; t[8] = encode["rel"][l[7]]
        t[9] = encode["race"][l[8]]; t[10] = encode["sex"][l[9]]; t[14] = encode["nc"][l[13]]

        # Binarize the numerical attributes based on median.
        # Modify this section to read the file in part c where you split the continuos attributes baed on dynamic median values.
        t[1] = float(l[0])
        t[3] = float(l[2])
        t[5] = float(l[4])
        t[11] = float(l[10])
        t[12] = float(l[11])
        t[13] = float(l[12])

        # Convert some of the booleans to ints
        data.append([int(x) for x in t])

    return np.array(data,dtype=np.int64)

def init():
    print('\r\nPREPROCESSING STARTED...')
    ## Read the data
    train_data = preprocess("dtree_data/train.csv")
    valid_data = preprocess("dtree_data/valid.csv")
    test_data = preprocess("dtree_data/test.csv")

    print("The sizes are ","Train:",len(train_data),", Validation:",len(valid_data),", Test:",len(test_data))
    print('PREPROCESSING DONE!\r\n')
    return train_data,valid_data,test_data,hash_map