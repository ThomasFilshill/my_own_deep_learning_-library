from neuronix.neural_network import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

from neuronix.neural_network import NeuralNet
import numpy as np

class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, net: NeuralNet) -> None:
        self.t += 1
        for idx, (param, grad) in enumerate(net.params_and_grads()):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(grad)
                self.v[idx] = np.zeros_like(grad)
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for parameter, gradient in net.params_and_grads():
            parameter -= self.lr * gradient


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, net: NeuralNet) -> None:
        for idx, (param, grad) in enumerate(net.params_and_grads()):
            if idx not in self.velocities:
                self.velocities[idx] = np.zeros_like(param)
            self.velocities[idx] = self.momentum * self.velocities[idx] - self.lr * grad
            param += self.velocities[idx]
