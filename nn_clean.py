import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as t
import sklearn.metrics as skm


def get_data(file_train, file_test):
    train_data_temp = pd.read_csv(file_train, header=None)
    test_data_temp = pd.read_csv(file_test, header=None)

    suit = {1: 'hearts', 2: 'spades', 3: 'diamonds', 4: 'clubs'}
    card = {1: 'Ace', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: '10', 11: 'jack', 12: 'queen', 13: 'king'}

    train_features = pd.DataFrame()
    test_features = pd.DataFrame()
    # train_features = train_data_temp.iloc[:,:-1]
    # test_features = test_data_temp.iloc[:,:-1]

    train_labels = train_data_temp.iloc[:, -1]
    test_labels = test_data_temp.iloc[:, -1]

    f.open('dump_of_data.txt', 'a')
    
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
                test_features[label] = (test_data_temp[i] == k).astype(int)
            for k in card:
                label = str(str(card_num) + '_' + str(k))
                train_features[label] = (train_data_temp[i+1] == k).astype(int)
                test_features[label] = (test_data_temp[i+1] == k).astype(int)

    f.write(train_features)
    f.write(train_labels)
    f.write(test_features)
    f.write(test_labels)

    return train_features, train_labels, test_features, test_labels


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
        # for 0 the derivative will be zero
        return (X > 0).astype(int)


def feedforward(n_w, n_b, input, activation_type, layers):

    lin_layer = [0 for i in range(len(layers))]
    act_layer = [0 for i in range(len(layers))]

    # first layer
    lin_layer[0] = (n_w[0] @ input) + n_b[0]
    act_layer[0] = activation(lin_layer[0], activation_type)

    # middle layers (but the last)
    for l in range(1, len(layers)-1):
        lin_layer[l] = (n_w[l] @ act_layer[l-1]) + n_b[l]
        act_layer[l] = activation(lin_layer[l], activation_type)
    
    # last layer (activation type to be sigmoid, always)
    l = len(layers) - 1
    lin_layer[l] = (n_w[l] @ act_layer[l-1]) + n_b[l]
    act_layer[l] = activation(lin_layer[l], 1)    

    return lin_layer, act_layer

def del_computation(n_w, lin_layer, act_last, output_instance, layers, activation_type):
      # ######----COMPUTE DEL------#######
      # initialize list for all layers 
      del_layer = [0 for i in range(len(layers))]

      # #### Last (output) layer
      # activation for the last layer will always be sigmoid 
      
      del_layer[-1] = (act_last - output_instance) * activation_diff(lin_layer[-1], 1)

      # #### Hidden layers
      for l in range(len(layers)-2, -1, -1):
          del_layer[l] = ((n_w[l+1].T)@(del_layer[l+1])) * activation_diff(lin_layer[l], activation_type)

      # for the entire batch you need to make the backprop for every example
      # and then using the average of all the del values in the batch update the weights and biases.
      # adjust weights
      # make a function for back prop on one example.
      # for using entire mini batch take average of del along rows
      # adjust weights for learning rate alpha, eta whatever
      # the del and activations must be of required dimensions and not already averaged ()np.mean(axis = 1, keepdims=True) )!
      # averaging should be done here using npmean

      return del_layer

def backprop(network_weights, network_bias, input_instance, output_instance):
    # Add doc string
    return 0


def gnn(batch_size, input_size, layers, output_size, features, labels, activation_type, learning_rate, iterations, tol=-1):

    layers.append(output_size)
    lr = learning_rate

    network_weights = [np.random.random((layers[0], input_size))]
    network_bias = [np.random.random((layers[0], 1))]
    for i in range(1, len(layers)):
        network_weights.append(np.random.random((layers[i], layers[i-1])))
        network_bias.append(np.random.random((layers[i], 1)))

    # FOR CHECKING PURPOSES # create randomly initialized weight matrix for each layer
    # network_weights = [np.zeros((layers[0], input_size))]
    # network_bias = [np.zeros((layers[0], 1))]
    # for i in range(1, len(layers)):
    #     network_weights.append(np.zeros((layers[i], layers[i-1])))
    #     network_bias.append(np.zeros((layers[i], 1)))

    training_error = []
    for iter in range(iterations):
        
        number_of_mini_batches = int(np.ceil(features.shape[0] / batch_size))
        batch_train_error = []
        
        for mb in range(number_of_mini_batches):
            input_instance = features[mb * batch_size:(mb + 1) * batch_size].T
            raw = labels[mb * batch_size:(mb + 1) * batch_size]
            output_instance = np.zeros((output_size, raw.shape[0]))
            for i in range(raw.shape[0]):
                output_instance[raw[i][0]][i] = 1

            # ################-----FEED FORWARD-----##########################

            linearity_layer, activation_layer = feedforward(network_weights, 
                                                            network_bias, 
                                                            input_instance, 
                                                            activation_type, 
                                                            layers)

            del_layer = del_computation(network_weights, 
                                        linearity_layer, 
                                        activation_layer[-1], 
                                        output_instance, 
                                        layers, 
                                        activation_type)
            
            diff = activation_layer[-1] - output_instance
            all_error = np.sum(0.5 * ((diff) ** 2), axis=0, keepdims=True)
            cost_minibatch = np.mean(all_error)
            # import math
            # if(math.isnan(cost_minibatch)):
            #     print('NAN found!!')
            #     breakpoint
            #     break        
            # cost_minibatch = np.mean(np.sum(0.5 * (activation_layer[-1] - output_instance) ** 2, axis=0, keepdims=True))
            # training_error.append(cost_minibatch)

            batch_train_error.append(cost_minibatch)
            x = (del_layer[0] @ input_instance.T)
            
            # logy = np.log10(np.max(x))
            # if logy <= -3:
            #     lr = np.power(10, -logy-2)
            # else:
            #     lr = learning_rate

            nab_w = (x * lr / del_layer[0].shape[1])
            nab_b = (lr * np.mean(del_layer[0], axis=1, keepdims=True))

            network_weights[0] = network_weights[0] + nab_w
            network_bias[0] = network_bias[0] + nab_b

            # network_weights[0] = network_weights[0] - ((del_layer[0] @ input_instance.T) * lr / del_layer[0].shape[1])
            # network_bias[0] = network_bias[0] - (lr * np.mean(del_layer[0], axis=1, keepdims=True))
            for l in range(1, len(layers)):
                    
                x = (del_layer[l] @ activation_layer[l -1].T)
                # logy = np.log10(np.max(x))
                # if logy <= -3:
                #     lr = np.power(10, -logy-2)
                # else:
                #     lr = learning_rate

                nab_w = (lr * x / del_layer[l].shape[1])
                nab_b = (lr * np.mean(del_layer[l], axis=1, keepdims=True))

                network_weights[l] = network_weights[l] + nab_w
                network_bias[l] = network_bias[l] + nab_b

                # network_weights[l] = network_weights[l] - (lr * (del_layer[l] @ activation_layer[l -1].T)/ del_layer[l].shape[1])
                # network_bias[l] = network_bias[l] - (lr * np.mean(del_layer[l], axis=1, keepdims=True))
            print(f'iter = {iter}, batch# = {mb}, average error = {cost_minibatch}')
        training_error.append(np.mean(batch_train_error))
        print(f'###################--Train Error: {iter} = {training_error[-1]}')
        if(tol != -1 and len(training_error) >=2 and training_error[-1] - training_error[-2] < tol):
            lr /= 5

    return network_weights, network_bias, training_error


def nn_prediciton(input_instance, n_w, n_b, activation_type):
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
    activation_layer[0] = activation(linearity_layer[0], activation_type)

    for l in range(1, len(n_w)-1):
        linearity_layer[l] = n_w[l] @ activation_layer[l-1] + n_b[l]
        activation_layer[l] = activation(linearity_layer[l], activation_type)
    
    l = len(n_w)-1
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
    '''[summary]
    
    Arguments:
        p {[type]} -- [description]
        test_labels {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''

    count = 0
    for i in range(len(p)):
        if test_labels[i][0] == p[i]:
            count += 1 

    return count*100.0/len(p)


def part_c(file_trian, file_test, epochs, tolerance, activation_type):

    # number of iterations 
    # iter = 1000

    train_features, train_labels, test_features, test_labels = get_data(file_train, file_test)
    # preprocessing the labels before sending!
    test_labels_array = np.array(test_labels).reshape((test_labels.shape[0], 1))
    train_labels_array = np.array(train_labels).reshape((train_labels.shape[0], 1))
    
    test_features_array = np.array(test_features)
    train_features_array = np.array(train_features)
    
    hidden_layer_size = [5,10,15,20,25]
    batch_size = 100
    input_layer = 85
    output_layer = 10
    activation_type = 1
    learning_rate = 0.1

    accuracy_list_test = []
    accuracy_list_train = []
    error_list = []
    time_list = []
    confusion_matrix_list = []
    for h in hidden_layer_size:
        start = t.time()
        # n_weights, n_bias, err = gnn(25010,
        #                                 85,
        #                                 [h],
        #                                 10,
        #                                 train_features_array,
        #                                 train_labels_array,
        #                                 activation_type,
        #                                 0.1,
        #                                 iter,
        #                                 tol=tolerance)


        n_weights, n_bias, err = gnn(batch_size, input_layer, [h], 
                                        output_layer, train_features_array, 
                                        train_labels_array, activation_type,
                                        learning_rate, epochs, tol=tolerance)
        end = t.time()

        p = nn_prediciton(test_features_array.T, n_weights, n_bias, activation_type)
        p1 = nn_prediciton(train_features_array.T, n_weights, n_bias, activation_type)

        acc_test = accuracy(p, test_labels_array)
        acc_train = accuracy(p1, train_labels_array)
        
        accuracy_list_test.append(p)
        accuracy_list_train.append(p1)
        time_list.append(end-start)
        # error_list.append(err)
        print(len(list(test_labels_array.flat)))
        confusion_matrix_list.append(skm.confusion_matrix(list(test_labels_array.flat), p))
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
    for i in range(len(hidden_layer_size)):
        print(f'Confusion Matrix for Hidden Layer Size = {hidden_layer_size}\n{confusion_matrix_list[i]}\n')


def part_d(file_trian, file_test, iter, tolerance, activation_type):

    # number of iterations 
    # iter = 1000

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
    confusion_matrix_list = []
    for h in hidden_layer_size:
        start = t.time()
        n_weights, n_bias, err = gnn(25010,
                                        85,
                                        [h,h],
                                        10,
                                        train_features_array,
                                        train_labels_array,
                                        activation_type,
                                        0.1,
                                        iter,
                                        tol=tolerance)
        end = t.time()

        p = nn_prediciton(test_features_array.T, n_weights, n_bias)
        p1 = nn_prediciton(train_features_array.T, n_weights, n_bias)

        acc_test = accuracy(p, test_labels_array)
        acc_train = accuracy(p1, train_labels_array)
        
        accuracy_list_test.append(p)
        accuracy_list_train.append(p1)
        time_list.append(end-start)
        error_list.append(err)
        confusion_matrix_list.append(skm.confusion_matrix(list(test_labels_array.flat), p))
    # Plot Test Accuracy
    x = hidden_layer_size
    y = accuracy_list_test
    plt.plot(x, y, color='g')
    plt.xlabel('Size of Hidden Layers')
    plt.ylabel('Accuracy on Test')
    plt.title('Accuracy vs LayerSIze')
    plt.show()
    plt.clf()
    # Plot Train Accuracy
    x = hidden_layer_size
    y = accuracy_list_train
    plt.plot(x, y, color='g')
    plt.xlabel('Size of Hidden Layers')
    plt.ylabel('Accuracy on Test')
    plt.title('Accuracy vs LayerSIze')
    plt.show()
    plt.clf()
    # Plot Training Time
    x = hidden_layer_size
    y = time_list
    plt.plot(x, y, color='g')
    plt.xlabel('Size of Hidden Layers')
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
    for i in range(len(hidden_layer_size)):
        print(f'Confusion Matrix for Hidden Layer Size = {hidden_layer_size}\n{confusion_matrix_list[i]}\n')


def part_e(file_trian, file_test, tol, activation_type):

    print('### PART C with adaptive learning rate ###')
    part_c(file_train, file_test, tol)
    print('### PART D with adaptive learning rate ###')
    part_d(file_train, file_test, tol)


def part_f(file_train, file_test, tol):
    part_e(file_train, file_test)
    
    return


# ######----MAIN----#######

file_train = './PokerDataset/poker-hand-training-true.data'
file_test = './PokerDataset/poker-hand-testing.data'
# file_train = './toy_train.csv'
# file_test = './toy_train.csv'

batch_size = 100
input_layer = 85
hidden_layer = [20]
output_layer = 10
epochs = 2000
activation_type = 1
learning_rate = 10

# batch_size = 100
# input_layer = 3
# hidden_layer = [7]
# output_layer = 8
# epochs = 10000
# activation_type = 1
# learning_rate = 0.5


# for adaptive learning rate, add tol=RATE as a param

train_features, train_labels, test_features, test_labels = get_data(file_train, file_test)

# preprocessing the labels before sending!
test_labels_array = np.array(test_labels).reshape((test_labels.shape[0], 1))
train_labels_array = np.array(train_labels).reshape((train_labels.shape[0], 1))
test_features_array = np.array(test_features)
train_features_array = np.array(train_features)

# print(test_features_array)
# print(test_labels_array)

n_weights, n_bias, err = gnn(batch_size, input_layer, hidden_layer, 
                                output_layer, train_features_array, 
                                train_labels_array, activation_type,
                                learning_rate, epochs, tol=0.001 )

# print(f'Weights = {n_weights}')
# print(f'Biases = {n_bias}')
# print(len(n_bias), len(n_weights))

# inst = test_features_array[0:1].T
# inst = test_features_array.T
# print(f'shape of inst = {inst.shape}')


x = range(0, len(err))
y = err
plt.plot(x, y, color='g')
# plt.plot(year, pop_india, color='orange')
plt.xlabel('#iterations')
plt.ylabel('Average Error')
plt.title('error vs iter')
plt.show()
plt.clf()
# plt.show()

# p = nn_prediciton(inst, n_weights, n_bias)
p = nn_prediciton(test_features_array.T, n_weights, n_bias, activation_type)
p1 = nn_prediciton(train_features_array.T, n_weights, n_bias, activation_type)

print(f'Test prediction set = {set(p)}')
print(f'Train prediction set = {set(p1)}')
print('Accuracy test = ' + str(accuracy(p, test_labels_array)))
print('Accuracy train= ' + str(accuracy(p1, train_labels_array)))

# part_c(file_train, file_test, epochs, -1, 1)

# part_c(file_train, file_test, epochs, 1e-4, 1)