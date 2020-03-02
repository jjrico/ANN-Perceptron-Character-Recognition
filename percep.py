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
def mat2np_arr_TRAIN(data):
    set = []
    for elem in data:
        row = char2vec(elem) # convert 5x7 dot matrix to linear vector
        row.insert(0,1) # preppend each row with a 1 for w0
        row.append(0) # append with a 0 for expected ouput
        set.append(row)

    return np.array(set) #convert to np array and return

# convert entire dataset to np array with leading 1 for w0
# DOES NOT append a 0 for expected input
def mat2np_arr_TEST(data):
    set = []
    for elem in data:
        row = char2vec(elem) # convert 5x7 dot matrix to linear vector
        row.insert(0,1) # preppend each row with a 1 for w0
        set.append(row)

    return np.array(set) #convert to np array and return

class Perceptron:
    # initializer
    # matrix of inputs must have leading ones and expected output values
    def __init__(self, input_mat, name, thresh = 0.0):
        self.input_mat = np.copy(input_mat)
        # initialize random weights
        # length of T_SET - 1 to account for expected output column
        # numbers range from -1 to 1
        self.weights = 2 * np.random.random_sample(len(self.input_mat[0]) - 1) - 1
        self.name = name
        self.thres = thresh # default threshold set to 0 for activation function

    # activation function used by train() and test()
    def activation_func(self, inputs):
        # find dot product of inputs and weights
        total_activation = np.dot(inputs, self.weights)
        # if total activation is greater than threshold, predict 1
        # otherwise predict 0
        return 1.0 if total_activation >= self.thres else 0.0

    # accuracy function to calculate the percentage of correct predictions made
    # by the model. Used by the train() to stop early if 100% accuracy has been
    # achieved. Only to be used by train() function
    # predictions calculated by the weights are created and stored here in preds
    def accuracy(self):
        num_correct = 0.0
        self.preds = []
        # for each letter in training set
        for i in range(len(self.input_mat)):
            # take prediction with input values (all values from input matrix
            # EXCEPT expected output) and current weights
            pred = self.activation_func(self.input_mat[i][:-1])
            self.preds.append(pred) # store predictions
            # if prediction matches expected output, increment number correct
            if pred == self.input_mat[i][-1]: num_correct += 1.0
        # output percentage of correct predictions made
        return num_correct/float(len(self.input_mat))

    # function used to train the perceptron
    # Default values:
    # Maximum # of epochs: 20
    # Learning Rate (eta): 1.00
    def train(self, max_epoch=20, l_rate=1.0, stop_early=True, do_print = True):

        if do_print: print('\nTRAINING PERCEPTRON: %s' % self.name)
        self.errors_made = []
        for epoch in range(max_epoch):
            num_errors = 0
            # calculate new predictions and accuracy
            # store predictions in self.preds
            cur_acc = self.accuracy()

            # if we have reached 100% accuracy we can stop early
            if cur_acc == 1.0 and stop_early:
                self.errors_made.append(num_errors)
                if do_print:
                    print("\nEpoch %d \nErrors made: %d" % (epoch + 1, num_errors))
                    print("Accuracy: %0.5f" %cur_acc)
                    print("Weights: ", self.weights)
                    print("Predictions: ", self.preds)
                break

            for i in range(len(self.input_mat)):
                # calculate error (+1 or -1 mean wrong prediction; 0 means correct)
                error = self.input_mat[i][-1] - self.preds[i]
                # if prediction was wrong increment number of errors
                if error != 0: num_errors += 1

                # update weights using eta(?) equation
                for j in range(len(self.weights)):
                    self.weights[j] += l_rate * error * self.input_mat[i][j]

            self.errors_made.append(num_errors)
            if do_print:
                print("\nEpoch %d \nErrors made: %d" % (epoch + 1, num_errors))
                print("Accuracy: %0.5f" % cur_acc)
                print("Weights: ", self.weights)
                print("Predictions: ", self.preds)

    # test function works with only one letter at a time
    def test(self, test_matrix):
        return self.activation_func(test_matrix)


    # plot error function
    def plot_error(self):
        epochs = np.arange(0, len(self.errors_made))
        plt.figure()
        plt.plot(epochs, self.errors_made, 'r+')
        plt.title("Training Error for " + self.name)
        plt.xlabel("EPOCH")
        plt.ylabel("# of Errors Made")
        plt.show()

# mat2np_arr prepends a 1 and appends a 0 to each column of the training data
TRAINING_SET = mat2np_arr_TRAIN(dataset.TRAINING_DATA)
TEST_SET = mat2np_arr_TEST(dataset.TEST_DATA)
dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
dict_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

perceptrons = []
"""
# one perceptron at a time for debugging
# non linear (??) D, G, J, M, N, Q, R
case = dict['Z']
TRAIN_SET = np.copy(TRAINING_SET)
TRAIN_SET[case][-1] = 1
p = Perceptron(TRAIN_SET, dict_[case])
p.train() # set do_print to true for errors and accuracy
p.plot_error()

for i in range(len(TEST_SET)):
    prediction = p.test(TEST_SET[i])
    print("Input: %s; Prediction: %d" % (dict_[i], prediction))


"""


# TRAINING
# Trains all perceptrons and appends them to a list
for i in range(len(TRAINING_SET)):
    TRAIN_SET = np.copy(TRAINING_SET)
    TRAIN_SET[i][-1] = 1
    p = Perceptron(TRAIN_SET, dict_[i])
    p.train(do_print = False) # set do_print to true for errors and accuracy
    p.plot_error()
    perceptrons.append(p)

# TESTING
# Tests every perceptron against every test character
# Prints message if the perceptron fires
for i in range(len(perceptrons)):
    predictions = []
    print("\nTesting perceptron: %s" % perceptrons[i].name)

    for j in range(len(TEST_SET)):
        prediction = perceptrons[i].test(TEST_SET[j])
        if prediction == 1:
            print("Activation!")
            print("Input: %s; Prediction: %d" % (dict_[j], prediction))

    #print(predictions)

#perceptrons[dict['Z']].plot_error()
