from random import randrange
from random import random
from csv import reader
from math import exp
import math
import matplotlib.pyplot as plt 


no_of_folds = 4 
# color corresponding to each fold
color = ['orange','red','green','yellow']

learning_rate = 0.5  # Learning rate
no_of_epochs = 500  
n_hidden = 5  # number of neurons in hidden layers
network = []


def initialize_weights(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer_1 = []
    for i in range(n_hidden):
        list = []
        dict = {}
        for j in range(n_inputs+1):
            list.append(random())
        dict['weights'] = list
        hidden_layer_1.append(dict)
    network.append(hidden_layer_1)


    hidden_layer_2 = []
    for i in range(n_hidden):
        list = []
        dict = {}
        for j in range(n_hidden+1):
            list.append(random())
        dict['weights'] = list
        hidden_layer_2.append(dict)
    network.append(hidden_layer_2)

    output_layer = []
    for i in range(n_outputs):
        list = []
        dict = {}
        for j in range(n_hidden+1):
            list.append(random())
        dict['weights'] = list
        output_layer.append(dict)
    network.append(output_layer)
    return network


# Find the min and max values for each column
def min_and_max(dataset):
    minmax = list()
    for column in zip(*dataset):
        minmax.append([min(column), max(column)])
    return minmax

 
# Split a dataset into k folds
def split_data(dataset, no_of_folds):
    dataset_split = []
    copy = list(dataset)
    fold_size = int(len(dataset) / no_of_folds)
    for i in range(no_of_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(copy))
            fold.append(copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Convert string column to float
def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Rescale dataset columns to the range 0-1 using z-factor algorithm
def normalize_dataset(dataset):
    for i in range(len(dataset[0])-1):
	    str_to_float(dataset, i)
    str_to_int(dataset, len(dataset[0])-1)
    min_max_list = min_and_max(dataset)
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])


 
# Calculate accuracy percentage
def accuracy_function(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def calculate_scores(dataset,n_folds, l_rate, no_of_epochs, n_hidden):

    # k-fold cross validation
    folds = split_data(dataset, n_folds)
    scores = list()
    fold_index = 0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = back_propagation(train_set,fold_index,test_set, l_rate, no_of_epochs, n_hidden)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_function(actual, predicted)
        scores.append(accuracy)
        fold_index = fold_index + 1
    
    return scores
 
# Calculate neuron activation for an input
def activate_function(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
 
# Transfer neuron activation
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate_function(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
       
# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
 

def training_the_dataset(network, train, l_rate, n_epoch, n_outputs,fold):
    
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += math.sqrt(sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]))
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            

        print('Fold=%d,Epoch %d Completed--->, error=%.3f' % (fold,epoch, sum_error))
        plt.scatter(epoch, sum_error,color=color[fold],label='Fold'+ str(fold))
        plt.pause(0.000001) 
# Initialize a network

 
# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train,fold ,test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = []
    network = initialize_weights(n_inputs, n_hidden, n_outputs)
    training_the_dataset(network, train, l_rate, n_epoch, n_outputs,fold)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)
 
# Load a CSV file in rows
def read_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# main 
dataset = read_csv('poker_data.csv')

# Normalizing the dataset
normalize_dataset(dataset)

# Labeling the graph
plt.xlabel('Epochs') 
plt.ylabel('Mean Squared Error')

prediction_percentage = calculate_scores(dataset, no_of_folds, learning_rate, no_of_epochs, n_hidden)
print('Prediction Percentage: %s' % prediction_percentage)
print('Mean Accuracy: %.3f%%' % (sum(prediction_percentage)/float(len(prediction_percentage))))
plt.show()
