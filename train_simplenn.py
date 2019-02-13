import numpy as np
import process
from pre_process import process_inputs, to_categorical, sigmoid
import matplotlib.pyplot as plt
import time
#import time


learning_rate = 5e-2
num_hidden_neurons = 25
num_outputs = 9
epochs = 2000
fraction = 0.2  # fraction of images used for test

# we get a 400x900 input matrix
inputs, outputs = process.getdata(sizex=30, sizey=30)
num_images = len(outputs)
print num_images
all_ids = range(num_images)
# print all_ids.shape
test_ids = np.random.randint(low=0, high=len(
    outputs), size=(int(len(outputs) * fraction)))
# print test_ids
train_ids = (list(set(all_ids) - set(test_ids)))

train_data = inputs[train_ids, :]
train_output = outputs[train_ids]

test_data = inputs[test_ids, :]
test_output = outputs[test_ids]

n_records = len(train_output)
n_features = inputs.shape[1]  # features are along rows

inputs, means, stds = process_inputs(train_data)
outputs = to_categorical(train_output, num_categories=num_outputs)
test_o_category = to_categorical(test_output, num_categories=num_outputs)


def predict(input_images, wt_i_h, wt_h_o):  # suppose N images are given: Nx900
    # names=['Jaiyam','Hikaru']
    input_images = (input_images - means) / stds

    hiddenl_i = np.dot(input_images, wt_i_h)
    hiddenl_o = sigmoid(hiddenl_i)  # produce a Nx10 matrix whose
    #[i,j] represents output for image i from neuron j
    outputl_i = np.dot(hiddenl_o, wt_h_o)
    probabilities = sigmoid(outputl_i)  # produces a Nx2 output
    probabilities /= np.sum(probabilities, axis=1)[:, None]
    predictions = np.argmax(probabilities, axis=1)
    # print predictions.shape
    return predictions


# We have an input layer with 900 neurons
# A hidden layer with 10 neurons
# An output layer with 2 neurons

# We need weights from input to hidden_layer
# which will be (900,10)
# Weights from hidden to output layer will be 10x2
# wij input i to next layer j

weights_i_h = np.random.normal(
    scale=1.0 / (n_features**0.5), size=(n_features, num_hidden_neurons))
weights_h_o = np.random.normal(
    scale=1.0 / (n_features**0.5), size=(num_hidden_neurons, num_outputs))

last_loss = []
train_acc = []
test_acc = []
train_err = []
test_err = []
# go forward through network, calculate output
# Get output error
# Get error_term for output layer del0
# Get error for hidden layer
# Get error_term for hidden layer del1
# go backward through network update weights

fig, ax = plt.subplots()
plt.xlim([0, epochs])
# plt.ion()
# plt.show()

for epoch in range(epochs):
    hiddenlayer_i = np.dot(inputs, weights_i_h)
    hiddenlayer_o = sigmoid(hiddenlayer_i)  # produce a 400x10 matrix whose
    #[i,j] represents output for image i from neuron j
    outputlayer_i = np.dot(hiddenlayer_o, weights_h_o)
    network_o = sigmoid(outputlayer_i)  # produces a 400x2 output

    errors = outputs - network_o  # 400x2 matrix

    del0 = errors * network_o * (1 - network_o)  # produces a 400x2 output
    # The above operation * is equivalent to np.multiply and is an element-wise multiplication
    # Matrix multiplication is done by np.dot() or np.matmul()
    # we want this to return a 400x10 matrix
    del1 = hiddenlayer_o * (1 - hiddenlayer_o) * np.dot(del0, weights_h_o.T)
    # details of derivation of this form are in written in notebook (physical
    # nb)

    delw_h_o = learning_rate * hiddenlayer_o.T.dot(del0) / n_records
    delw_i_h = learning_rate * inputs.T.dot(del1) / n_records

    weights_i_h += delw_i_h
    weights_h_o += delw_h_o
    last_loss.append(np.mean(errors**2))
    if epoch % (epochs / 500) == 0:
        # this_step=epoch/(epochs/50)
        test_predictions = predict(test_data, weights_i_h, weights_h_o)
        test_accuracy = np.sum(
            (test_output == test_predictions)) * 100 / float(fraction * num_images)
        test_predictions = to_categorical(
            test_predictions, num_categories=num_outputs)

        train_predictions = to_categorical(
            np.argmax(network_o, axis=1), num_categories=num_outputs)
        training_accuracy = np.sum(
            train_output == train_predictions) * 100 / float(n_records)

        training_error = np.mean((outputs - network_o)**2)
        testing_error = np.mean((test_predictions - test_o_category)**2)

        train_err.append(training_error)
        test_err.append(testing_error)
        #plt.plot(range(epoch, epochs/10),)

        # print train_acc[0]

        print 'Test accuracy = ', test_accuracy
        # time.sleep(1)
    # print(np.mean(errors**2))
    # print np.mean(errors)
ax.plot(epochs * np.arange(500) / 500, train_err,
        color='r', label='Training error')
ax.plot(epochs * np.arange(500) / 500, test_err, color='b', label='Test error')
#plt.figlegend((line1,line2),('CPUtemp','GPUtemp'),'upper right')
legend = ax.legend(loc='best')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error')
plt.savefig('train_test.png')
plt.show()
# plt.pause(0.001)


np.save('weights_i_h' + str(num_hidden_neurons) + '.npy', weights_i_h)
np.save('weights_h_o' + str(num_hidden_neurons) + '.npy', weights_h_o)
np.save('means.npy', means)
np.save('stds.npy', stds)
fig2, ax2 = plt.subplots()
ax2.plot(np.log(np.array(last_loss)))
plt.xlabel('Epochs')
plt.ylabel('log(Loss)')
plt.savefig('loss' + str(num_hidden_neurons) + '.png')
plt.show()
predictions = network_o / np.sum(network_o, axis=1)[:, None]
print predictions
predictions = predictions > 0.5

training_accuracy = np.sum(
    (outputs == predictions) / float(num_outputs)) * 100 / float(n_records)
print 'Training Accuracy = ', training_accuracy, '%'
