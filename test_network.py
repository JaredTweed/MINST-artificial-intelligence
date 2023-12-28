import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)





# Testing the network
#### Modify the code to get the confusion matrix ####
# all_preds = []
# for i in range(0, xtest.shape[1], 100):
#     cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
from sklearn.metrics import confusion_matrix

# Initialize predictions list and true label list
all_preds = []
all_labels = []

# Testing the network
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
    # Store the predictions and true labels
    all_preds.extend(np.argmax(P, axis=0))
    all_labels.extend(ytest[0, i:i+100])

# Compute and print the confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", conf_mat)

# Find the top confused pairs of classes
confused_pairs = []
for i in range(len(conf_mat)):
    for j in range(i + 1, len(conf_mat)):
        # conf_mat[i, j] = the number of times i is predicted to be j
        confused_pairs.append(((i, j), conf_mat[i, j] + conf_mat[j, i]))
confused_pairs.sort(key=lambda x: x[1], reverse=True)

# Print the top two most commonly confused pairs
print("Top two confused pairs:")
for i, ((class_a, class_b), count) in enumerate(confused_pairs[:2]):
    print(f"{i + 1}. Class {class_a} and class {class_b} have been confused {count} times.")

# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)

