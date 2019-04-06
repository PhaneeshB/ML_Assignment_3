import numpy as np
import pandas as pd
import copy as c
from scipy import stats
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# a fn
# give column and dataset
# return probs of unique ele


def compute_attr_prob(table, col_index):
    num_vals = set(table[:, col_index])
    dic_count = dict()
    for i in num_vals:
        dic_count[i] = 0

    for i in table[:,col_index]:
        dic_count[i] += 1

    total = len(list(table[:, col_index]))

    for k in dic_count:
        dic_count[k] /= total

    return dic_count

# fn given matrix 
# find entropy 


def get_entropy(table):
    unique_ele_prob = compute_attr_prob(table, -1)
    sum=0
    for i in unique_ele_prob:
        sum += -unique_ele_prob[i] * np.log2(unique_ele_prob[i])
    return sum

# given a matrix partition it into 
# matrices of same unique 


def divide(table, col_index):
    unique_ele = set(table[:, col_index])
    col = table[:,col_index]
    dic_parts = dict()
    for i in unique_ele:
        dic_parts[i] = table[col==i]
    return dic_parts


def info_gain(table, col_index, attr_entropy):
    
    parts = divide(table, col_index)
    probs = compute_attr_prob(table, col_index)
    broken_ent = 0
    for i in probs:
        ent = get_entropy(parts[i])
        broken_ent += probs[i] * ent
    return attr_entropy - broken_ent


def get_best_attribute(table, entropy):

    max_infogain_attr_index = 0
    max_infogain_value = info_gain(table, 0, entropy)
    # all but the last column
    for i in range(1, table.shape[1] -1 ):
        inf_gain = info_gain(table, i, entropy)

        if (inf_gain > max_infogain_value):
            max_infogain_value = inf_gain
            max_infogain_attr_index = i

    return max_infogain_value, max_infogain_attr_index


class tree_node:
    def __init__(self, attr_num=-1, attr_ent=0, isLeaf=False, descendants_dict=None, median_list = []):
        self.attribute_num = attr_num
        self.node_entropy = attr_ent
        self.leaf = isLeaf
        self.prediction = -1
        self.descendants = descendants_dict
        self.median_list = median_list


def create_tree(root, table):

    entropy = get_entropy(table)
    root.prediction = stats.mode(table[:, -1])
    root.node_entropy = entropy
    if entropy == 0:
        root.leaf = True
        root.descendants = None
        root.attribute_num = -1
        return

    else:
        split_info_gain, split_attribute = get_best_attribute(table, entropy)

        # split_info_gain = info_gain(table, split_attribute, split_attribute_entropy)
        if (split_info_gain < 1e-6):
            root.leaf = True
            root.descendants = None
            root.attribute_num = split_attribute
            return
        # this implies the node has to be an internal nodes
        else:
            root.descendants = dict()
            root.attribute_num = split_attribute
            
            descendants_data = divide(table, split_attribute)
            for k in descendants_data:
                root.descendants[k] = tree_node()
                create_tree(root.descendants[k], descendants_data[k])

            return



def divide_partc(table, table_copy, col_index):
    
    unique_ele = set(table_copy[:, col_index])
    col = table_copy[:,col_index]
    dic_parts = dict()
    for i in unique_ele:
        dic_parts[i] = table[col==i]
    return dic_parts



def get_best_attribute_partc(table_og, entropy):

    table_copy = c.deepcopy(table_og)
    format_attr_index = [0,4]
    format_attr_index.extend(list(range(11,23)))
    median_list = []
    for i in format_attr_index:
        med = np.median(table_copy[:, i])
        table_copy[:, i] = (table_copy[:, i] > med).astype(int)
        median_list.append(med)

    max_infogain_attr_index = 0
    max_infogain_value = info_gain(table_copy, 0, entropy)

    # all but the last column
    for i in range(1, table_copy.shape[1] -1 ):
        inf_gain = info_gain(table_copy, i, entropy)

        if (inf_gain > max_infogain_value):
            max_infogain_value = inf_gain
            max_infogain_attr_index = i

    return max_infogain_value, max_infogain_attr_index, median_list, table_copy

def create_tree_partc(root, table_og):

    entropy = get_entropy(table_og)
    root.prediction = stats.mode(table_og[:, -1])
    root.node_entropy = entropy

    if entropy == 0:
        root.leaf = True
        root.descendants = None
        root.attribute_num = -1
        return

    else:
        
        split_info_gain, split_attribute, med_list, table_copy = get_best_attribute_partc(table_og, entropy)

        # split_info_gain = info_gain(table_og, split_attribute, split_attribute_entropy)
        if (split_info_gain < 1e-6):
            root.leaf = True
            root.descendants = None
            root.attribute_num = split_attribute
            return
        # this implies the node has to be an internal nodes
        else:
            root.descendants = dict()
            root.attribute_num = split_attribute
            root.median_list = med_list
            descendants_data = divide_partc(table_og, table_copy, split_attribute)
            for k in descendants_data:
                root.descendants[k] = tree_node()
                create_tree_partc(root.descendants[k], descendants_data[k])

            return

def get_label(node, row):
    if node.leaf:
        return node.prediction
    else:
        children = node.descendants
        n = node.attribute_num
        key = row[n]
        if key not in list(children.keys()):
            return node.prediction
        else:
            new_node = children[row[n]]
            # print(f'Node =  {node.attribute_num}  row[n] =  {row[n]}  child = {node.descendants[row[n]].attribute_num}')
            return get_label(new_node, row)
        # get_label(node.descendants[row[node.attribute_num]], row)


def prediction(root, test_data):
    pred_class = []
    true_class = []
    for i in test_data:
        # print(f'Row = {i}')
        label = get_label(root, i)
        pred_class.append(label[0])
        true_class.append(i[-1])

    count = 0
    # print(pred_class[0])
    for i in range(len(true_class)):
        if true_class[i] == pred_class[i]:
            count += 1

    return pred_class, (count *100.0/len(true_class))


def get_label_partc(node, row):

    format_attr_index = [0,4]
    format_attr_index.extend(list(range(11,23)))
    if node.leaf:
        return node.prediction
    
    else:

        children = node.descendants
        n = node.attribute_num
        key = row[n]

        if n in format_attr_index:
            if key > node.median_list[format_attr_index.index(n)]:
                key = 1
            else:
                key = 0
            
            new_node = children[key]
            # print(f'Node =  {node.attribute_num}  row[n] =  {row[n]}  child = {node.descendants[row[n]].attribute_num}')
            return get_label_partc(new_node, row)

        else:    
            if key not in list(children.keys()):
                return node.prediction
            else:
                new_node = children[row[n]]
                # print(f'Node =  {node.attribute_num}  row[n] =  {row[n]}  child = {node.descendants[row[n]].attribute_num}')
                return get_label_partc(new_node, row)
        # get_label(node.descendants[row[node.attribute_num]], row)

def prediction_partc(root, test_data):

    pred_class = []
    true_class = []
    for i in test_data:
        label = get_label_partc(root, i)
        pred_class.append(label[0])
        true_class.append(i[-1])

    count = 0
    # print(pred_class[0])
    for i in range(len(true_class)):
        if true_class[i] == pred_class[i]:
            count += 1

    return pred_class, (count *100.0/len(true_class))


# ######----MAIN-----#########

# file_train = './dataset/credit-cards.train.csv'
# file_test = './dataset/credit-cards.test.csv'
# file_val = './dataset/credit-cards.val.csv'

# train_data  = pd.read_csv(file_train, skiprows = 1)
# test_data = pd.read_csv(file_test, skiprows = 1)
# val_data = pd.read_csv(file_val, skiprows = 1)

# #separate features and labels
# # info about the attributes can be found here: 
# # https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#
# #clean the data currently custom designed for this dataset
# # continuous attrs = 1(amt),5(age), 12 - 23

# format_attr_index = [1,5]
# format_attr_index.extend(list(range(12,24)))

# for i in format_attr_index:
#     med = train_data.iloc[:, i].median()
#     train_data.iloc[:, i] = (train_data.iloc[:, i] > med).astype(int)
#     test_data.iloc[:, i] = (test_data.iloc[:, i] > med).astype(int)
#     val_data.iloc[:, i] = (val_data.iloc[:, i] > med).astype(int)
#     # print(train_data.iloc[:, i])

# train_data = train_data.iloc[:, 1:]
# test_data = test_data.iloc[:, 1:]
# val_data = val_data.iloc[:, 1:]

# print(train_data[0:10])

# root = tree_node()
# create_tree(root, np.array(train_data))

# print(root)
# print(prediction(root, np.array(test_data))[1])
# print(prediction(root, np.array(train_data))[1])


####### PART C 


file_train = './dataset/credit-cards.train.csv'
file_test = './dataset/credit-cards.test.csv'
file_val = './dataset/credit-cards.val.csv'

train_data  = pd.read_csv(file_train, skiprows = 1)
test_data = pd.read_csv(file_test, skiprows = 1)
val_data = pd.read_csv(file_val, skiprows = 1)


#separate features and labels
# info about the attributes can be found here: 
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#
#clean the data currently custom designed for this dataset
# continuous attrs = 1(amt),5(age), 12 - 23


train_data = train_data.iloc[:, 1:]
test_data = test_data.iloc[:, 1:]
val_data = val_data.iloc[:, 1:]


root_c = tree_node()
create_tree_partc(root_c, np.array(train_data))
print(root_c)

print(prediction_partc(root_c, np.array(test_data))[1])
print(prediction_partc(root_c, np.array(train_data))[1])


def part_d():
    
    file_train = './dataset/credit-cards.train.csv'
    file_test = './dataset/credit-cards.test.csv'
    file_val = './dataset/credit-cards.val.csv'

    train_data  = pd.read_csv(file_train, skiprows = 1)
    test_data = pd.read_csv(file_test, skiprows = 1)
    val_data = pd.read_csv(file_val, skiprows = 1)


    model = tree.DecisionTreeClassifier()
    model.fit(train_data.iloc[:,1:-1],train_data.iloc[:,-1])
    pred_class=model.predict(train_data.iloc[:,1:-1])
    print ("The training accuracy is ",acc(pred_class,train_data.iloc[:,-1]))

    pred_class=model.predict(val_data.iloc[:,1:-1])
    print ("The validation accuracy is ", acc(pred_class,val_data.iloc[:,-1]))
    
    pred_class=model.predict(test_data.iloc[:,1:-1])
    print( "The testing accuracy is ",acc(pred_class,test_data.iloc[:,-1]))


    # x=
    # y=
    # plot(x,y)
part_d()

def acc(pred_class, true_class):
    
    count = 0
    for i in range(len(true_class)):
        if true_class[i] == pred_class[i]:
            count += 1

    return (count *100.0/len(true_class))

def part_f():
    
    file_train = './dataset/credit-cards.train.csv'
    file_test = './dataset/credit-cards.test.csv'
    file_val = './dataset/credit-cards.val.csv'

    train_data  = pd.read_csv(file_train, skiprows = 1)
    test_data = pd.read_csv(file_test, skiprows = 1)
    val_data = pd.read_csv(file_val, skiprows = 1)

    model = RandomForestClassifier()
    model.fit(train_data.iloc[:,1:-1],train_data.iloc[:,-1])
    pred_class=model.predict(train_data.iloc[:,1:-1])
    print ("The training accuracy is ",acc(pred_class,train_data.iloc[:,-1]))

    pred_class=model.predict(val_data.iloc[:,1:-1])
    print ("The validation accuracy is ", acc(pred_class,val_data.iloc[:,-1]))
    
    pred_class=model.predict(test_data.iloc[:,1:-1])
    print( "The testing accuracy is ",acc(pred_class,test_data.iloc[:,-1]))

    # x=
    # y=
    # plot(x,y)
    
part_f()


def part_e():
    train = onehot_decision()


def plot(x,y):

    plt.plot(x, y, color='g')
    # plt.plot(year, pop_india, color='orange')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.show()
    plt.clf()
    # plt.show()

def onehot_decision():

    file_train = './dataset/credit-cards.train.csv'
    file_test = './dataset/credit-cards.test.csv'
    file_val = './dataset/credit-cards.val.csv'

    train_data  = pd.read_csv(file_train, skiprows = 1)
    test_data = pd.read_csv(file_test, skiprows = 1)
    val_data = pd.read_csv(file_val, skiprows = 1)

    train_data_final = pd.DataFrame()
    format_attr_index = [2,3,4,6,7,8,9,10,11]

    attr_name = [list(train_data.columns)[i] for i in format_attr_index]

    unique_values =  {'SEX': [1, 2],
    'EDUCATION': [0, 1, 2, 3, 4, 5, 6],
    'MARRIAGE': [0, 1, 2, 3],
    'PAY_0':[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'PAY_2':[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'PAY_3':[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'PAY_4':[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'PAY_5':[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'PAY_6':[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    for col in train_data.columns:
        
        if col in attr_name:
            # add multiple columns 
            print(col)
            for i in range(len(unique_values[col])):
                label = col + '_' + str(unique_values[col][i])
                train_data_final[label] = (train_data[col] == unique_values[col][i]).astype(int)
                test_data_final[label] = (test_data[col] == unique_values[col][i]).astype(int)
                val_data_final[label] = (t_data[col] == unique_values[col][i]).astype(int)

        else:
            # add the column as it is 
            train_data_final[col] = train_data[col]
            train_data_final[col] = train_data[col]
            train_data_final[col] = train_data[col]

return train_data_final
