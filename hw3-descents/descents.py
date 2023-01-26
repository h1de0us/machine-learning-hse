from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        pred = self.predict(x)
        if self.loss_function is LossFunction.MSE:
            return np.sum((pred - y) ** 2) / y.shape[0]
        if self.loss_function is LossFunction.LogCosh:
            return np.sum(np.log(np.cosh(pred - y))) / y.shape[0]
        if self.loss_function is LossFunction.MAE:
            return np.sum(np.abs(pred - y)) / y.shape[0]
        if self.loss_function is LossFunction.Huber:
            if np.linalg.norm(y - pred) <= 1.35: # magic constant
                return 1/2 * np.linalg.norm(pred - y)
            else:
                return 1.35 * np.sum(np.abs(y - pred - 1.35 / 2)) / y.shape[0]
        
        

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        pred = x.dot(self.w)
        return pred
        


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        # TODO: implement updating weights function
#         print('update weights')
        delta = self.lr() * gradient
#         print(delta.shape, self.w.shape)
        self.w = self.w - delta
        return -delta
        

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # TODO: implement calculating gradient function
        pred = self.predict(x)
        if self.loss_function is LossFunction.MSE:
            grad = -2 * x.T.dot(y - pred) / y.shape[0]
        if self.loss_function is LossFunction.LogCosh:
            grad = x.T.dot(np.tanh(pred - y)) / y.shape[0]
        if self.loss_function is LossFunction.MAE:
            grad = x.T.dot(np.sign((pred - y))) / y.shape[0]
        if self.loss_function is LossFunction.Huber:
            if np.linalg.norm(y - pred) <= 1.35: # magic constant
                grad = -x.T.dot(y - pred) / y.shape[0]
            else:
                grad = - 1.35 * x.T.dot(np.sign(y - pred)) / y.shape[0]
            
        return grad



class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_indexes = np.random.randint(x.shape[0], size=self.batch_size)
        pred = self.predict(x)
#         print(x.shape, y.shape, pred.shape)
#         print(type(x), type(y), type(pred))
        y_batch = y[batch_indexes]
        pred_batch = pred[batch_indexes]
        x_batch = x[batch_indexes]
#         gradient = -2 * (y_batch - pred_batch).T.dot(x_batch)
        if self.loss_function is LossFunction.MSE:
            gradient = -2 * x_batch.T.dot(y_batch - pred_batch)
        if self.loss_function is LossFunction.LogCosh:
            gradient = x_batch.T.dot(np.tanh(pred_batch - y_batch))
        if self.loss_function is LossFunction.MAE:
            gradient = x_batch.T.dot(np.sign((pred_batch - y_batch)))
        if self.loss_function is LossFunction.Huber:
            if np.linalg.norm(y - pred) <= 1.35: # magic constant
                gradient = -x_batch.T.dot(y_batch - pred_batch)
            else:
                gradient = - 1.35 * x_batch.T.dot(np.sign(y_batch - pred_batch))
        return gradient / self.batch_size


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w = self.w - self.h
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        self.iteration += 1
        mhat = self.m / (1 - self.beta_1 ** self.iteration)
        vhat = self.v / (1 - self.beta_2 ** self.iteration)
        delta = self.lr() / (np.sqrt(vhat) + self.eps) * mhat
        self.w -= delta
        return -delta
    
class AdaMax(VanillaGradientDescent):
    """
    It is a variant of Adam based on the infinity norm.
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
            
        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0
            
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = np.maximum(self.beta_2 * self.v, np.abs(gradient))
        self.iteration += 1
        delta = self.lr() / (1 - self.beta_1 ** self.iteration) * self.m / (self.v + self.eps)
        self.w -= delta
        return delta
        
        

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient = self.w # TODO: replace with L2 gradient calculation
        l2_gradient[-1] = 0

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'adamax': AdaMax
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
