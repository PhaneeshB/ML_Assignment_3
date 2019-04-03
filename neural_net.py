import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as t
# part a One hot encoding
# find out multivalued attributes
# for every value of the multivalued attribute add a column in the data set
# (Assuming that all possible values of the attribute
# will always be in the training data)


def get_data(file_train, file_test):
    train_data_temp = pd.read_csv(file_train, header=None)
    test_data_temp = pd.read_csv(file_test, header=None)

    suit = {1: 'hearts', 2: 'spades', 3: 'diamonds', 4: 'clubs'}
    card = {1: 'Ace', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: '10', 11: 'jack', 12: 'queen', 13: 'king'}

    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    train_labels = train_data_temp.iloc[:, -1]
    test_labels = test_data_temp.iloc[:, -1]

    # can be made more readable
    for i in train_data_temp.columns:
        if i == 10:
            break
        if i % 2 == 0:
            card_num = int(i/2 + 1)
            print('i = ' + str(i))
            for k in suit:
                label = str(str(card_num) + '_' + suit[k])
                train_features[label] = (train_data_temp[i] == k).astype(int)
                test_features[label] = (train_data_temp[i] == k).astype(int)
            for k in card:
                label = str(str(card_num) + '_' + str(k))
                train_features[label] = (test_data_temp[i+1] == k).astype(int)
                test_features[label] = (test_data_temp[i+1] == k).astype(int)

    return train_features, train_labels, test_features, test_labels


# print(test_features[0:10])
# print(test_features.columns)
# print(len(test_features.columns))
# 33
# part B :Generic neural network
# params: Batch Size for SGD, #inputs, #&size of hidden layers, #outputs
# fully connected architecture
# sigmoid activation unit


def activation(X, type=1):
    # type 1 = sigmoid
    # type 2 = ReLU
    if type == 1:
        return 1/(1 + np.exp(-X))
    elif type == 2:
        return np.maximum(X, 0)


def activation_diff(X, code=1):
    '''Activation differentiation wrt their arguments.

    Arguments:
        X {[type]} -- Activation ndarray to be differentiated

    Keyword Arguments:
        code {int} -- To select the type of activation used (default: {1})

    Returns:
        ndarray -- processed ndarray of the same dimensions as argument
    '''
    # code 1 = sigmoid
    # code 2 = ReLU
    if code == 1:
        return np.exp(X)/((1 + np.exp(X))**2)
    elif code == 2:
        return (X >= 0).astype(int)
# Takes input as a slice of features and labels
# returns


def backprop(network_weights, network_bias, input_instance, output_instance):
    # Add doc string
    return 0


def gnn(batch_size, input_size, layers, output_size, features, labels, learning_rate, iterations):

    # output layer is part of the layers.s
    layers.append(output_size)
    # # create randomly initialized weight matrix for each layer
    # network_weights = [np.random.random((layers[0], input_size))]
    # network_bias = [np.random.random((layers[0], 1))]

    # for i in range(1, len(layers)):
    #     network_weights.append(np.random.random((layers[i], layers[i-1])))
    #     network_bias.append(np.random.random((layers[i], 1)))

    # FOR CHECKING PURPOSES # create randomly initialized weight matrix for each layer
    network_weights = [np.zeros((layers[0], input_size))]
    network_bias = [np.zeros((layers[0], 1))]

    for i in range(1, len(layers)):
        network_weights.append(np.zeros((layers[i], layers[i-1])))
        network_bias.append(np.zeros((layers[i], 1)))

    # compute output for a given input, i.e., the activation for the last layer

    # input_instance = np.zeros((input_size, 1))  # temporary features
    # output_instance = np.zeros(output_size) # temporary label

    # to run the entire data set for 100 iterations 
    training_error = []
    for iter in range(iterations):
        number_of_mini_batches = int(np.ceil(features.shape[0] / batch_size))
        for mb in range(number_of_mini_batches):
            # no need to check because python doesn't give errors
            # if((i+1) * batch_size < features.shape[0]):
            
            input_instance = features[mb * batch_size:(mb + 1) * batch_size].T

            # output instance must be preprocessed to match the activation in the last layer!
            raw = labels[mb * batch_size:(mb + 1) * batch_size]
            output_instance = np.zeros((output_size, raw.shape[0]))
            for i in range(raw.shape[0]):
                output_instance[raw[i][0]][i] = 1
            # for i in raw
            # no need to iterate over each example.
            # we use matirx operations for the entire batch
            # for ex in range(batch_size):

            # just lists with required size
            linearity_layer = [np.zeros(1) for i in range(len(layers))]
            activation_layer = [np.zeros(1) for i in range(len(layers))]

            # feedforward
            linearity_layer[0] = (network_weights[0] @ input_instance) + network_bias[0]
            activation_layer[0] = activation(linearity_layer[0], 1)

            for l in range(1, len(layers)):
                linearity_layer[l] = (network_weights[l] @ activation_layer[l-1]) + network_bias[l]
                activation_layer[l] = activation(linearity_layer[l], 1)

            # final output 0-9 + 1
            # print("prediction = " + str(np.argmax(activation_layer[-1], axis=0)))
            # print("label      = " + str(np.argmax(labels[-1], axis=0)))

            # compute del for output layer
            del_layer = [np.zeros((1, 1)) for i in range(len(layers))]
            del_layer[-1] = (activation_layer[-1] - output_instance) * activation_diff(linearity_layer[-1])

            # compute del for hidden layers
            for l in range(len(layers)-2, -1, -1):
                del_layer[l] = ((network_weights[l+1].T)@(del_layer[l+1])) * activation_diff(linearity_layer[l])
            # for the entire batch you need to make the backprop for every example
            # and then using the average of all the del values in the batch update the weights and biases.

            # adjust weights
            # make a function for back prop on one example.

            # for using entire mini batch take average of del along rows

            # adjust weights for learning rate alpha, eta whatever
            # the del and activations must be of required dimensions and not already averaged ()np.mean(axis = 1, keepdims=True) )!
            # averaging should be done here using npmean

            # adjust weights
            lr = learning_rate
            # prod = del_layer[0] @ input_instance.T
            # bs = del_layer[0].shape[1]
            # network_weights[0] = network_weights[0] - (lr * (prod)/bs)
            network_weights[0] = network_weights[0] - ((del_layer[0] @ input_instance.T) * lr / del_layer[0].shape[1])
            network_bias[0] = network_bias[0] - (lr * np.mean(del_layer[0], axis=1, keepdims=True))
            for l in range(1, len(layers)):
                network_weights[l] = network_weights[l] - (lr * (del_layer[l] @ activation_layer[l -1].T)/ del_layer[l].shape[1])
                network_bias[l] = network_bias[l] - (lr * np.mean(del_layer[l], axis=1, keepdims=True))
            
            # avg cost or error in prediction for the entire minibatch
            # same as taking mean over the entire array
            cost_minibatch = np.mean(np.sum(0.5 * (activation_layer[-1] - output_instance) ** 2, axis=0, keepdims=True))
            training_error.append(cost_minibatch)
            print(f'iter = {iter}, batch# = {mb}, average error = {cost_minibatch}')

    # final cost of prediction
    # final output 0-9 + 1
    # print("prediction = " + str(np.argmax(activation_layer[-1] + 1)))
    # repeat
    return network_weights, network_bias, training_error


def nn_prediciton(input_instance, n_w, n_b):
    '''This method is used for calculating the output of a
    neural network for a given instance.
    Arguments:
        input_instance {ndarray} -- this is supposed to be a vector (n,1)
        but can also be a series of vectors (n,m)
        n_w {list of ndarrays} -- list of weight matrices for
        all the layers in the network.
        n_b {list of ndarrays} -- list of biases for all the neurons

    Returns:
        int/list of ints -- the output of the network for
        single/multiple instance of input
    '''
    # Here we only need to store the output of the current layer 
    # and the previous layer. (but we're all of it)
    linearity_layer = [np.zeros(1) for i in range(len(n_w))]
    activation_layer = [np.zeros(1) for i in range(len(n_w))]
    
    # feedforward
    linearity_layer[0] = n_w[0] @ input_instance + n_b[0]
    activation_layer[0] = activation(linearity_layer[0], 1)

    for l in range(1, len(n_w)):
        linearity_layer[l] = n_w[l] @ activation_layer[l-1] + n_b[l]
        activation_layer[l] = activation(linearity_layer[l], 1)

    # final output 0-9 for single instance of input
    if input_instance.shape[1] == 1:
        prediction = np.argmax(activation_layer[-1])
    else:
        x = np.argmax(activation_layer[-1], axis=0)
        prediction = list(x)

    return prediction


def accuracy(p, test_labels):
    count = 0
    for i in range(len(p)):
        if test_labels[i][0] == p[i]:
            count += 1 

    return count*100.0/len(p)


def part_c(file_trian, file_test):

    # number of iterations 
    iter = 1000

    file_train = './PokerDataset/poker-hand-training-true.data'
    file_test = './PokerDataset/poker-hand-testing.data'
    train_features, train_labels, test_features, test_labels = get_data(file_train, file_test)
    # preprocessing the labels before sending!
    test_labels_array = np.array(test_labels).reshape((test_labels.shape[0], 1))
    train_labels_array = np.array(train_labels).reshape((train_labels.shape[0], 1))
    test_features_array = np.array(test_features)
    train_features_array = np.array(train_features)
    hidden_layer_size = [5,10,15,20,25]
    accuracy_list_test = []
    accuracy_list_train = []
    error_list = []
    time_list = []
    for h in hidden_layer_size:
        start = t.time()
        n_weights, n_bias, err = gnn(25010,
                                        85,
                                        [h],
                                        10,
                                        train_features_array,
                                        train_labels_array,
                                        0.1,
                                        iter)
        end = t.time()

        p = nn_prediciton(test_features_array.T, n_weights, n_bias)
        p1 = nn_prediciton(train_features_array.T, n_weights, n_bias)
        accuracy_list_test.append(p)
        accuracy_list_train.append(p1)
        time_list.append(end-start)
        error_list.append(err)
    # Plot Test Accuracy
    x = hidden_layer_size
    y = accuracy_list_test
    plt.plot(x, y, color='g')
    plt.xlabel('#HiddenLayers')
    plt.ylabel('Accuracy on Test')
    plt.title('Accuracy vs LayerSIze')
    plt.show()
    plt.clf()
    # Plot Train Accuracy
    x = hidden_layer_size
    y = accuracy_list_train
    plt.plot(x, y, color='g')
    plt.xlabel('#HiddenLayers')
    plt.ylabel('Accuracy on Test')
    plt.title('Accuracy vs LayerSIze')
    plt.show()
    plt.clf()
    # Plot Training Time
    x = hidden_layer_size
    y = time_list
    plt.plot(x, y, color='g')
    plt.xlabel('#HiddenLayers')
    plt.ylabel('Accuracy on Test')
    plt.title('Accuracy vs LayerSIze')
    plt.show()
    plt.clf()
    # Plot Error Time
    x = hidden_layer_size
    y = error_list
    graph_colour = ['b', 'g', 'y', 'c', 'm', 'r']
    for i in range(len(y)):
        plt.plot(x, y[i], color=graph_colour[i])
    plt.xlabel('#Iterations')
    plt.ylabel('Average cost per batch')
    plt.title('Cost vs Iter (For Different Hidden Layers)')
    plt.show()

    # confusion matrix 


# ######----MAIN----#######

file_train = './PokerDataset/poker-hand-training-true.data'
file_test = './PokerDataset/poker-hand-testing.data'

train_features, train_labels, test_features, test_labels = get_data(file_train, file_test)

# preprocessing the labels before sending!
test_labels_array = np.array(test_labels).reshape((test_labels.shape[0], 1))
train_labels_array = np.array(train_labels).reshape((train_labels.shape[0], 1))
test_features_array = np.array(test_features)
train_features_array = np.array(train_features)


n_weights, n_bias, err = gnn(25010, 
                                85, 
                                [30], 
                                10, 
                                train_features_array, 
                                train_labels_array, 
                                0.1, 
                                10)

print(len(n_bias), len(n_weights))

# inst = test_features_array[0:1].T
inst = test_features_array.T
print(f'shape of inst = {inst.shape}')


x = range(0, len(err))
y = err
plt.plot(x, y, color='g')
# plt.plot(year, pop_india, color='orange')
plt.xlabel('#iterations')
plt.ylabel('Average Error')
plt.title('error vs iter')
plt.show()
plt.clf()
plt.show()

#p = nn_prediciton(inst, n_weights, n_bias)
p = nn_prediciton(inst, n_weights, n_bias)
p1 = nn_prediciton(train_features_array.T, n_weights, n_bias)

# print(f'prediction = {p}')
print('Accuracy test = ' + str(accuracy(p, test_labels_array)))
print('Accuracy train= ' + str(accuracy(p1, train_labels_array)))
