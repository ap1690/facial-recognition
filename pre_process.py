import numpy as np


def process_inputs(inputs):
    # inputs is a 2D matrix containing features along rows
    # and different observations along columns
    # inputs[i,:] is one observation
    # inputs[:,i] is one feature for all obs
    # returns (inputs-mean)/stds bw -1 and 1

    means = np.mean(inputs, axis=0)  # mean of each feature
    stds = np.std(inputs, axis=0)  # std deviation of each feature
    norm_inputs = (inputs - means) / (stds)

    return [norm_inputs, means, stds]


def to_categorical(outputs, num_categories=2):
    # creates a one-hot encoding of outputs
    result = []
    for output in outputs:
        # print output
        categorical = np.zeros(num_categories)
        categorical[output] = 1
        result.append(categorical)

    # Hopefully this will produce a 400x2 matrix
    result = np.reshape(result, (len(outputs), num_categories))
    return result


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x))
