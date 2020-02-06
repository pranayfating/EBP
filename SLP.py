from perceptron import Perceptron

BLUE = 1
RED = 0
LMSE = 0.001

def normalise(data):
    temp_list = []
    for entry in data:
        entry_list = []
        for value in entry[0]:
            entry_list.append(float(value*0.003921568))
        temp_list.append([entry_list, entry[1]])
    return temp_list

def main(data):

    training_data = normalise(data)

    p = Perceptron(len(data[0][0]))

    epochs = 0
    mse =999

    while (abs(mse-LMSE) > 0.002):

        error = 0

        for value in training_data:
            output = p.result(value[0])
            iter_error = value[1] - output
            error += iter_error
            p.weight_adjustment(value[0], iter_error)
        mse = float(error/len(training_data))

        # Print the MSE for each epoch
        print( "The MSE of %d epochs is %.10f" % (epochs, mse))

        # Every 100 epochs show the weight values
        if epochs % 100 == 0:
            print( "0: %.10f - 1: %.10f - 2: %.10f - 3: %.10f" % (p.w[0], p.w[1], p.w[2], p.w[3]))

        # Increment the epoch number
        epochs += 1

    return p