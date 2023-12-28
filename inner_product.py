import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape # returns dimensions
    n = param["w"].shape[1] # returns depth

    ###### Fill in the code here ######
    # print("INNER START")
    # print(param["w"].shape) # (800, 500)
    # print(input["data"].shape) # (800, 100)
    # print(param["b"].shape) # (500, 500)

    # print()
    # # print(input["w"].T.shape)
    # # print(param["data"].shape)
    # print()
    # print(np.dot(input["data"].T, param["w"]).shape)    
    # print(param["b"].shape)
    # print()
    # print(n)
    # print(k)
    # print((np.dot(input["data"].T, param["w"]) + param["b"]).T.shape)
    # print("INNER END")
    
    output_data = np.matmul(param["w"].T, input["data"]) + param["b"].T
    



    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": output_data # np.zero((n,k)) #  # replace 'data' value with your implementation
    }

    # print(output)

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    
    # param_grad['b'] = np.zeros_like(param['b'])
    # param_grad['w'] = np.zeros_like(param['w'])
    # input_od = None

    # Gradient with respect to weights
    param_grad['w'] = np.dot(output['diff'], input_data['data'].T).T
    
    # Gradient with respect to biases
    param_grad['b'] = np.sum(output['diff'], axis=1)
    
    # Gradient with respect to input data
    input_od = np.dot(param['w'], output['diff'])

    # print("INNER BACK START")
    # print(param_grad['w'].shape)
    # print(param_grad['b'].shape)
    # print(input_od.shape)
    # print("INNER BACK END")

    return param_grad, input_od
