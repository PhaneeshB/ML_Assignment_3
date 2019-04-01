import numpy as np
import pandas as pd

# part a: implement decision tree
## read the data
## clean the data (change continuous attribute to boolean)
## create entropy function
## create best attr function
## create write the rec algorithm
# 

# compute entropy
# given an attribute find the information gain when splitting the data set on that attribute.
#### sys arguments 
file_train = './dataset/credit-cards.train.csv'
file_test = './dataset/credit-cards.test.csv'
file_val = './dataset/credit-cards.val.csv'


train_data  = pd.read_csv(file_train, skiprows = 1)
test_data = pd.read_csv(file_test, skiprows = 1)
val_data = pd.read_csv(file_val, skiprows = 1)

#separate features and labels
# info about the attributes can be found here: 
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#
train_attr = train_data.iloc[:, :-1]
train_label = train_data.iloc[:, -1]
#clean the data currently custom designed for this dataset
# continuous attrs = 1(amt),5(age), 12 - 23

format_attr_index = [1,5]
format_attr_index.extend(list(range(12,24)))

for i in format_attr_index:
    med = train_attr.iloc[:, i].median()
    train_attr.iloc[:, i] = (train_attr.iloc[:, i] >= med).astype(int)
    print( train_attr.iloc[:, i])

#print(train_attr[0:10])

# calculate initial entropy without any split

