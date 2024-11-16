import numpy as np
import matplotlib.pyplot as plt
from neuronix.train import train
from neuronix.neural_network import NeuralNet
from neuronix.layers import Linear, Tanh
from neuronix.optimizers import SGD

x_values = np.linspace(0, 2 * np.pi, 1000)
inputs = x_values.reshape(-1, 1)
inputs = (inputs - np.pi) / np.pi  
targets = np.sin(x_values).reshape(-1, 1)

net = NeuralNet([
    Linear(input_size=1, output_size=16),
    Tanh(),
    Linear(input_size=16, output_size=16),
    Tanh(),
    Linear(input_size=16, output_size=1)
])

train(net, inputs, targets, num_epochs=1000, optimizer=SGD(lr=0.001))

test_input = np.array([[np.pi / 4]])
scaled_test_input = (test_input - np.pi) / np.pi 
predicted = net.forward(scaled_test_input)
print("Predicted sin(π/4):", predicted)
print("Actual sin(π/4):", np.sin(np.pi / 4))

predicted_targets = net.forward(inputs)

plt.plot(x_values, targets, label='Actual Sine Wave')
plt.plot(x_values, predicted_targets, label='Predicted Sine Wave', linestyle='dashed')
plt.legend()
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave Prediction')
plt.show()