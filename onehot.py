import pandas as pd
import numpy as np


def onehot_neural(file_train, file_test):

    train_data_temp = pd.read_csv(file_train, header=None)
    test_data_temp = pd.read_csv(file_test, header=None)

    suit = {1: 'hearts', 2: 'spades', 3: 'diamonds', 4: 'clubs'}
    card = {1: 'Ace', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: '10', 11: 'jack', 12: 'queen', 13: 'king'}

    train_features = pd.DataFrame()
    test_features = pd.DataFrame()
    train_features_file = pd.DataFrame()
    test_features_file = pd.DataFrame()
    train_features = train_data_temp.iloc[:,:-1]
    test_features = test_data_temp.iloc[:,:-1]

    train_labels = train_data_temp.iloc[:, -1]
    test_labels = test_data_temp.iloc[:, -1]

    # can be made more readable
    for i in train_data_temp.columns:
        if i == 10:
            break
        if i % 2 == 0:
            card_num = int(i/2 + 1)
            # print('i = ' + str(i))
            for k in suit:
                label = str(str(card_num) + '_' + suit[k])
                # train_features[label] = (train_data_temp[i] == k).astype(int)
                train_features_file[label] = (train_data_temp[i] == k).astype(int)
                # test_features[label] = (test_data_temp[i] == k).astype(int)
                test_features_file[label] = (test_data_temp[i] == k).astype(int)
            for k in card:
                label = str(str(card_num) + '_' + str(k))
                # train_features[label] = (train_data_temp[i+1] == k).astype(int)
                train_features_file[label] = (train_data_temp[i+1] == k).astype(int)
                # test_features[label] = (test_data_temp[i+1] == k).astype(int)
                test_features_file[label] = (test_data_temp[i+1] == k).astype(int)

    for i in range(10):
        train_features_file[str(i)] = (train_labels == i).astype(int)
        test_features_file[str(i)] = (test_labels == i).astype(int)

    # write to file train and test features file
    # return train_features, train_labels, test_features, test_labels
    return (train_features_file, test_features_file)


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

        else:
            # add the column as it is 
            train_data_final[col] = train_data[col]

    return train_data_final
