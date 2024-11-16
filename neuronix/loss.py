import numpy as np

from neuronix.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
    
    
class BinaryCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return (predicted - actual) / (predicted * (1 - predicted) * actual.shape[0])

class CategoricalCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.sum(actual * np.log(predicted)) / predicted.shape[0]

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return (predicted - actual) / predicted.shape[0]