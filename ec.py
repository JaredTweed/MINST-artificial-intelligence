import cv2
import numpy as np
from utils import get_lenet
from init_convnet import init_convnet
from conv_net import convnet_forward
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math

# Load the model architecture
layers = get_lenet(batch_size=1)
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']
for params_idx in range(len(params)):
    raw_w = params_raw[0, params_idx][0, 0][0]
    raw_b = params_raw[0, params_idx][0, 0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

image_paths = [
    '../images/image1.JPG',
    '../images/image2.JPG',
    '../images/image3.png',
    '../images/image4.JPG'
]

for image_path in image_paths:
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding & finding contours
    thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted_digits = []
    for contour in contours:
        # Identify digit
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 200: continue # Avoid detecting very small regions as digits.
        digit = gray_image[y:y+h, x:x+w]

        # Fix color
        normalized_digit = (digit - np.min(digit)) / (np.max(digit) - np.min(digit))
        if np.mean(digit) >= 0.5: inv_dig = 1 - normalized_digit 
        else: inv_dig = normalized_digit
        
        # Padding and resizing
        pad = 0.1
        height, width = digit.shape
        size = max(height, width)
        square_digit = np.zeros((size, size))
        y_off = (size - height) // 2
        x_off = (size - width) // 2
        square_digit[y_off:y_off + height, x_off:x_off + width] = inv_dig
        padded_digit = np.pad(square_digit, [(round(size*pad), ), (round(size*pad), )], mode='constant')
        resized_digit = cv2.resize(padded_digit, (28, 28)).reshape((28*28, 1))
        
        # Forward pass through the network and make predictions.
        _, probs = convnet_forward(params, layers, resized_digit, test=True)
        predicted_digit = np.argmax(probs)
        extracted_digits.append((digit, predicted_digit))
        




    # Displaying Results

    # Determine figure size
    if(len(extracted_digits) > 10): row_width = 20 
    else: row_width = 10
    orig_image_width = 8
    num_digits = len(extracted_digits)
    num_rows_for_digits = math.ceil(num_digits / row_width)
    total_rows = 1 + num_rows_for_digits
    plt.figure(figsize=(8, 2 * total_rows))

    # Place the original image
    orig_col_loc = (row_width - orig_image_width) // 2  # calculate starting column for original
    plt.subplot2grid((total_rows, row_width), (0, orig_col_loc), colspan=orig_image_width)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    # Display the other images
    for idx, (digit, label) in enumerate(extracted_digits):
        row = idx // row_width + 2  # Determine row for digit
        col = idx % row_width + 1  # Determine column for digit
        subplot_num = (row - 1) * row_width + col
        if subplot_num > total_rows * row_width:
            break  # Exit if subplot limit is reached
        plt.subplot(total_rows, row_width, subplot_num)
        plt.imshow(digit, cmap='gray')
        plt.title(f'{label}')
        plt.axis('off')

    plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
    plt.show()
