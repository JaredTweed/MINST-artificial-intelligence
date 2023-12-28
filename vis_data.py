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
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()

output = convnet_forward(params, layers, xtest[:,0:1])
output_1 = np.reshape(output[0]['data'], (28,28), order='F')

##### Fill in your code here to plot the features ######

def plot_features(features, title, height, width, num_channels, batch_size):
    for b in range(batch_size):
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"{title} - Batch {b + 1}")
        for i in range(min(20, num_channels)):
            plt.subplot(4, 5, i + 1)
            start_idx = i * height * width
            end_idx = start_idx + height * width
            feature_map = features[start_idx:end_idx, b]
            feature_map_reshaped = np.reshape(feature_map, (height, width), order='F')
            oriented_feature_map = np.flipud(np.fliplr(feature_map_reshaped).T)
            plt.imshow(oriented_feature_map, cmap='gray')
            plt.axis('off')
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        plt.savefig(f"../my_images/vis_data_results/{title}_Batch_{b + 1}.png")
        plt.show()

conv_output = output[1]['data']
conv_height = output[1]['height']
conv_width = output[1]['width']
conv_channels = output[1]['channel']
conv_batch_size = output[1]['batch_size']
plot_features(conv_output, 'CONV_Layer_Features', conv_height, conv_width, conv_channels, conv_batch_size)

relu_output = output[2]['data']
relu_height = output[2]['height']
relu_width = output[2]['width']
relu_channels = output[2]['channel']
relu_batch_size = output[2]['batch_size']
plot_features(relu_output, 'ReLU_Layer_Features', relu_height, relu_width, relu_channels, relu_batch_size)
