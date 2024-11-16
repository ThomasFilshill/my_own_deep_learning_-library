import numpy as np
import matplotlib.pyplot as plt
from neuronix.train import train
from neuronix.neural_network import NeuralNet
from neuronix.layers import Linear, Tanh
from neuronix.optimizers import SGD
from neuronix.loss import MSE

x_values = np.linspace(-10, 10, 200)
inputs = x_values.reshape(-1, 1)
targets = (x_values ** 2).reshape(-1, 1)

inputs_mean = np.mean(inputs)
inputs_std = np.std(inputs)
inputs = (inputs - inputs_mean) / inputs_std

targets_mean = np.mean(targets)
targets_std = np.std(targets)
targets = (targets - targets_mean) / targets_std

print("Inputs mean:", np.mean(inputs), "Inputs std:", np.std(inputs))
print("Targets mean:", np.mean(targets), "Targets std:", np.std(targets))

net = NeuralNet([
    Linear(input_size=1, output_size=32),  
    Tanh(),
    Linear(input_size=32, output_size=32),
    Tanh(),
    Linear(input_size=32, output_size=1)
])

optimizer = SGD(lr=0.001) 

train(
    net,
    inputs,
    targets,
    num_epochs=1000,
    loss=MSE(),
    optimizer=optimizer
)

predicted_targets = net.forward(inputs)

predicted_targets = predicted_targets * targets_std + targets_mean

plt.plot(x_values, x_values ** 2, label='Actual $y = x^2$')
plt.plot(x_values, predicted_targets, label='Predicted', linestyle='--')
plt.legend()
plt.title('Quadratic Function Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
