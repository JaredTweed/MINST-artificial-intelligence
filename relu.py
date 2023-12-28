import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    
    output['data'] = np.zeros_like(input_data['data'])
    output['data'] = np.maximum(0, input_data['data'])
    
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    
    # Creating a matrix with the same shape as input_data['data']
    input_od = np.zeros_like(input_data['data'])
    
    input_od = np.copy(output['diff'])
    input_od[input_data['data'] < 0] = 0


    # print(output)
    # print(layer)

    # For the indices where input_data['data'] is positive, input_od = output['diff'] 
    # input_od[input_data['data'] > 0] = output['diff'][input_data['data'] > 0]

    return input_od
