import numpy as np
import matplotlib.pyplot as plt
import dataset

# convert dot matrix to binary vector
def char2vec(char):
    return [
        0 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]

# convert entire dataset to np array with leading 1 for w0
# also appends 0 to each row later to be used for storing expected output
def mat2np_arr(data):
    set = []
    for elem in data:
        row = char2vec(elem) # convert 5x7 dot matrix to linear vector
        row.insert(0,1) # preppend each row with a 1 for w0
        row.append(0) # append with a 0 for expected ouput
        set.append(row)

    return np.array(set) #convert to np array and return

def plot(x, y, title, xaxis="EPOCH", yaxis="#  of Errors made"):
        plt.figure()
        plt.plot(x, y, 'r+')
        plt.title(title)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

# activation function
def activation_func(inputs, weights):
    thres = 0.0 # threshold set to 0
    # find dot product of inputs and weights
    total_activation = np.dot(inputs, weights)
    # if total activation is greater than threshold, predict 1
    # otherwise predict 0
    return 1.0 if total_activation >= thres else 0.0

# accuracy function to calculate the percentage of correct predictions made by
# the model. Used by the train_weights function to stop early if accuracy has
# reached 1.00
def accuracy(matrix, weights):
    num_correct = 0.0
    preds = []
    for i in range(len(matrix)):
        # take prediction with input values (all values from input matrix
        # EXCEPT expected output) and current weights
        pred = activation_func(matrix[i][:-1], weights)
        preds.append(pred) # store predictions
        # if prediction matches expected output, increment number correct
        if pred == matrix[i][-1]: num_correct += 1.0
    print("\nPredictions: ", preds)
    # output percentage of correct predictions made
    return num_correct/float(len(matrix))

# function used to train the perceptron
# Default values:
# Maximum # of epochs: 50
# Learning Rate (eta): 1.00
def train_weights(matrix, weights, title, max_epoch=50, l_rate=1.00, do_plot=False,
                  stop_early=True):

    errors_made = []
    for epoch in range(max_epoch):
        num_errors = 0
        num_epochs = epoch
        cur_acc = accuracy(matrix, weights) #calculate current accuracy
        print("\nEpoch %d \nWeights: " % epoch,weights)
        print("Accuracy: ", cur_acc)

        # if we have reached 100% accuracy we can stop early
        if cur_acc == 1.0 and stop_early:
            errors_made.append(num_errors)
            break
        #if do_plot: plot(matrix,weights,title="Epoch %d"%epoch)

        for i in range(len(matrix)):
            # predict output
            prediction = activation_func(matrix[i][:-1], weights)
            # calculate error (+1 or -1 mean wrong prediction; 0 means correct)
            error = matrix[i][-1] - prediction
            # if prediction was wrong increment number of errors
            if error != 0: num_errors += 1
            # update weights using eta(?) equation
            for j in range(len(weights)):
                weights[j] = weights[j] + (l_rate * error * matrix[i][j])
        print("Errors: ", num_errors)
        errors_made.append(num_errors)

    # plot errors by number of epochs
    epochs = np.arange(0, num_epochs + 1)
    plot(epochs, errors_made, title)

    return weights

TRAINING_SET = mat2np_arr(dataset.TRAINING_DATA)
#TEST_SET = mat2np_arr(dataset.TEST_DATA)
WEIGHTS = np.random.rand(len(TRAINING_SET[0]) - 1) # subtract 1 for input column
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i in range(len(TRAINING_SET)):
    title = "Perceptron Training Error: "
    TRAIN_SET = TRAINING_SET
    TRAIN_SET[i][-1] = 1
    print('\nLETTER: ', letters[i])
    train_weights(TRAIN_SET, WEIGHTS, title + letters[i])

plt.show()
