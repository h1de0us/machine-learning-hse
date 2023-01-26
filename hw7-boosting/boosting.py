from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import scipy


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        if self.base_model_params is not None:
            model = self.base_model_class(**self.base_model_params)
        else:
            model = self.base_model_class()
        
        idxs = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample), replace=True)
        subsample_x = x[idxs]
        subsample_y = y[idxs]
        model.fit(subsample_x, subsample_y)
        new_predictions = model.predict(x)
        self.gammas.append(self.find_optimal_gamma(y, predictions, new_predictions) * self.learning_rate)
        self.models.append(model)
                           
        return self
               

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        
        
        '''
        В fit приходит две выборки, обучающая и валидационная. На обучающей мы обучаем новые базовые модели,
        на валидационной считаем качество для ранней остановки (если это предусматривают параметры).

        Сначала нам нужно сделать какую-то нулевую модель, сделать предсказания для обучающей и валидационной выборок 
        (в шаблоне это нулевая модель, соответственно предсказания это просто np.zeros). 
        После этого нужно обучить n_estimators базовых моделей (как и на что обучаются базовые модели смотрите в лекциях и семинарах). 
        После каждой обученной базовой модели мы должны обновить текущие предсказания, 
        посчитать ошибку на обучающей и валидационной выборках (используем loss_fn для этого), проверить на раннюю остановку.

        После всего цикла обучения надо нарисовать график (если plot).
        '''
        
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        self.history['train'] = []
        self.history['val'] = []
        self.history['score'] = []

        for i in range(self.n_estimators):
#             новые базовые модели обучаются на ошибках

            shift_train = -self.loss_derivative(y_train, train_predictions)
            self.fit_new_base_model(x_train, shift_train, train_predictions)
            train_predictions += self.models[-1].predict(x_train) * self.gammas[-1]
            valid_predictions += self.models[-1].predict(x_valid) * self.gammas[-1]
            self.history['train'].append(self.loss_fn(y_train, train_predictions))
            self.history['val'].append(self.loss_fn(y_valid, valid_predictions))
            self.history['score'].append(self.score(x_valid, y_valid))
            
                              
            if self.early_stopping_rounds is not None and i < self.early_stopping_rounds:
                self.validation_loss[i] = self.loss_fn(y_valid, valid_predictions)
                if i != np.argmin(self.validation_loss):
                    break

        if self.plot:
            plt.plot(np.arange(self.n_estimators), self.history['train'], label='train_loss')
            plt.plot(np.arange(self.n_estimators), self.history['val'], label='val_loss')
            plt.legend()
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += gamma * model.predict(x)
        probas = self.sigmoid(predictions).reshape(-1, 1)
        return np.concatenate((1 - probas, probas), axis=1) 
            

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        importances = None
        for model in self.models:
            if importances is None:
                importances = model.feature_importances_
            else:
                importances += model.feature_importances_
        return importances / np.sum(importances)
