import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

# theta_1
weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

# theta_2
weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)    #z2 = x * theta_1
hidden_layer_output = sigmoid(hidden_layer_input)       #a2 = g(z2)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)    #z3 = a2 * theta_2
output = sigmoid(output_layer_in)                                       #a3 = g(z3)

## Backwards pass
## TODO: Calculate error: y - a_3, a_3 = g(z_3)
error = target - output

# TODO: Calculate error gradient for output layer: delta_3 = error .* g'(z_3)
del_err_output = error * output * (1.0 - output)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.multiply(hidden_layer_output, 1.0 - hidden_layer_output)
del_err_hidden = np.multiply(del_err_hidden, np.dot(del_err_output, weights_hidden_output))

print(del_err_output)
print(del_err_hidden)

# TODO: Calculate change in weights for hidden layer to output layer:
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_o = learnrate * np.dot(x[:,None], del_err_hidden[None,:])

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_o)
