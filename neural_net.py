import pandas as pd
import numpy as np

# part a One hot encoding 
## find out multivalued attributes 
## for every value of the multivalued attribute add a column in the data set 
## (Assuming that all possible values the attribute will always be in the training data)
file_train = './PokerDataset/poker-hand-training-true.data'
file_test = './PokerDataset/poker-hand-testing.data'

train_data_temp = pd.read_csv(file_train, header=None)
test_data_temp = pd.read_csv(file_test, header=None)

suit = {1:'hearts',2:'spades', 3:'diamonds', 4:'clubs' }
card = {1:'Ace', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',10:'10',11:'jack',12:'queen',13:'king'}

train_features = pd.DataFrame()
test_features = pd.DataFrame()

train_labels = train_data_temp.iloc[:, -1]
test_labels = test_data_temp.iloc[:, -1]

#can be made more readable
for i in train_data_temp.columns:
    if i == 10:
        break
    if i%2 == 0:
        card_num = int( i/2 + 1)
        print('i = ' + str(i))
        for k in suit:
            label = str(str(card_num) + '_' + suit[k])
            train_features[label] = (train_data_temp[i] == k).astype(int)
            test_features[label] = (train_data_temp[i] == k).astype(int)
        for k in card:
            label = str(str(card_num) + '_' + str(k))
            train_features[label] = (test_data_temp[i+1] == k).astype(int)
            test_features[label] = (test_data_temp[i+1] == k).astype(int)

# print(test_features[0:10])
# print(test_features.columns)
# print(len(test_features.columns))
###########################################################33
# part B :Generic neural network
# params: Batch Size for SGD, #inputs, #&size of hidden layers, #outputs
# fully connected architecture
# sigmoid activation unit


def activation(X, type=1):
    # type 1 = sigmoid
    # type 2 = ReLU
    if type == 1:
        return 1/(1 + np.exp(X))
    elif type == 2:
        return np.maximum(X, 0)

# needs to know the input
# type of activation function to be used
def gnn(batch_size, input_size, layers, output_size):
    
    # create randomly initialized weight matrix for each layer
    network_weights = [np.random.random((layers[0], input_size))]
    network_bias = [np.random.random((layers[0],1))]
    for i in range(1, len(layers)):
        network_weights.append(np.random.random((layers[i],layers[i-1])))
        network_bias.append(np.random.random((layers[i],1)))

    
    # compute output for a given input, i.e., the activation for the last layer
    
    input_instance = np.zeros((input_size,1)) # temporary input
    linearity_layer = [np.zeros(1) for i in range(len(layers))]
    activation_layer = [np.zeros(1) for i in range(len(layers))]
    
    linearity_layer[0] = network_weights[0] @ input_instance + network_bias[0]
    activation_layer[0] = activation(linearity_layer[0], 1)

    for l in range(1, len(layers)):
        linearity_layer[l] = network_weights[l] @ activation_layer[l-1] + network_bias[l]
        activation_layer[l] = activation(linearity_layer[l], 1)
        
    #final output 0-9 + 1
    print("prediction = " + str(np.argmax(activation_layer[-1] + 1)))
    # compute error for output layer
    output_instance = np.zeros(output_size)

    error_output_layer = np.sum((activation_layer[-1] - output_instance) ** 2)

    # compute del for output layer
    del_layer = [np.zeros((1,1)) for i in range(len(layers))]
    ## for the entire batch you need to make the backprop for every example 
    ## and then using the average of all the del values in the batch update the weights and biases.
    ## make a function for back prop on one example.

    # adjust weights
    # compute del for hidden layers
    # adjust weights 
    # repeat
    return 0