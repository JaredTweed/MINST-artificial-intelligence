import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# For image uploading
from PIL import Image

# Load the model architecture
layers = get_lenet(batch_size=1)
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

# Loop over each image
for i in range(1, 6): 
    image_path = f'../my_images/image_{i}.jpg'

    # Convert image to array
    image = Image.open(image_path)
    grayscale_image = image.convert('L')  # Convert to grayscale
    resized_image = grayscale_image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.asarray(resized_image) / 255.0 # Convert to normalized [0, 1] array
    if np.mean(image_array) >= 0.5: image_array = 1 - image_array  # Invert colors if necessary
    image_array = image_array.reshape(1, 28, 28, 1).transpose(1, 2, 3, 0) # Add a batch dimension
    
    # Get the predicted label
    cp, P = convnet_forward(params, layers, image_array, test=True)
    predicted_label = np.argmax(P, axis=0)[0]

    # Display the image and the prediction
    print(f'Image {i}:     True Label: {i}  Predicted Label: {predicted_label}')
    plt.subplot(1, 5, i)
    plt.imshow(image)
    # plt.imshow(image_array.reshape(28, 28), cmap='gray')
    plt.title(f'Image {i}\nPredicted: {predicted_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()