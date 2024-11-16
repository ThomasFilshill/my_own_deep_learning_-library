# circle_classification.py

import numpy as np
import matplotlib.pyplot as plt
from neuronix.train import train
from neuronix.neural_network import NeuralNet
from neuronix.layers import Linear, Tanh, Sigmoid
from neuronix.loss import BinaryCrossEntropy
from neuronix.optimizers import SGD

# Generate data
np.random.seed(0)
num_samples = 1000
radius = 5

inputs = np.random.uniform(-10, 10, (num_samples, 2))
targets = np.array([
    [1] if x[0]**2 + x[1]**2 < radius**2 else [0]
    for x in inputs
]).astype(np.float32)

# Define the network
net = NeuralNet([
    Linear(input_size=2, output_size=8),
    Tanh(),
    Linear(input_size=8, output_size=1),
    Sigmoid()
])

# Train the network
train(
    net,
    inputs,
    targets,
    num_epochs=1000,
    loss=BinaryCrossEntropy(),
    optimizer=SGD(lr=0.001)
)

# Test the network
test_inputs = inputs
predictions = net.forward(test_inputs)
predicted_classes = (predictions > 0.5).astype(int)

# Plot the results
plt.figure(figsize=(8, 8))
plt.scatter(inputs[:, 0], inputs[:, 1], c=predicted_classes.flatten(), cmap='bwr', alpha=0.5)
circle = plt.Circle((0, 0), radius, color='green', fill=False, linewidth=2)
plt.gca().add_artist(circle)
plt.title('Points Classified Inside or Outside the Circle')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
