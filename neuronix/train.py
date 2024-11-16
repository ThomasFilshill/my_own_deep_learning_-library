from neuronix.tensor import Tensor
from neuronix.neural_network import NeuralNet
from neuronix.loss import Loss, MSE
from neuronix.optimizers import Optimizer, SGD
from neuronix.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in iterator(inputs, targets):
            predictions = net.forward(batch.inputs)
            total_loss += loss.loss(predictions, batch.targets)
            gradients = loss.grad(predictions, batch.targets)
            net.backward(gradients)
            optimizer.step(net)
        print(epoch, total_loss)
