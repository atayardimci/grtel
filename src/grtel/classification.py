import numpy as np
import pandas as pd

from hottbox.algorithms.classification import TelVI
from hottbox.core import TensorTKD

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from beartype import beartype


class GRTEL:
    """Graph Regularized Tensor Ensemblef Learning"""
    @beartype
    def __init__(self,
        base_classifiers,
        n_classes: int = 1,
        probability: bool = False,
        verbose: bool = False,
    ):
        self.n_classes = n_classes
        self.probability = probability
        self.verbose = verbose
        self.models = [
            TelVI(
                base_clf=base_clf,
                probability=self.probability,
                verbose=self.verbose
            )
            for base_clf in base_classifiers
        ]

    @beartype
    def fit(self, X: list[TensorTKD], y: np.ndarray) -> None:
        for i in range(self.n_classes):
            print(i, end=" - ")
            self.models[i].fit(X, y[:, i])
        print()

    @beartype
    def score(self, X: list[TensorTKD], y: np.ndarray) -> list[float]:
        return [self.models[i].score(X, y[:, i]) for i in range(self.n_classes)]

    @beartype
    def grid_search(self, X: list[TensorTKD], y: np.ndarray, search_params) -> None:
        for i in range(self.n_classes):
            print(i, end=" - ")
            self.models[i].grid_search(X, y[:, i], search_params)
        print()

    @beartype
    def predict(self, X: list[TensorTKD]) -> list[np.ndarray]:
        return [
            self.models[i].predict(X)
            for i in range(self.n_classes)
        ]

    @beartype
    def confusion_matrices(self, X: list[TensorTKD], y: np.ndarray) -> list[np.ndarray]:
        predictions = self.predict(X)
        return [
            confusion_matrix(y[:, i], predictions[i])
            for i in range(self.n_classes)
        ]


class MultiClassifier:
    @beartype
    def __init__(self, n_classes: int = 1, verbose: bool = False):
        self.n_classes = n_classes
        self.verbose = verbose
        self.models = [
            DecisionTreeClassifier() for _ in range(n_classes)
        ]

    @beartype
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        for i in range(self.n_classes):
            print(i, end=" - ")
            self.models[i].fit(X, y[:, i])
        print()

    @beartype
    def score(self, X: pd.DataFrame, y: np.ndarray) -> list[float]:
        return [self.models[i].score(X, y[:, i]) for i in range(self.n_classes)]

    @beartype
    def predict(self, X: pd.DataFrame) -> list[np.ndarray]:
        return [
            self.models[i].predict(X)
            for i in range(self.n_classes)
        ]

    @beartype
    def confusion_matrices(self, X: pd.DataFrame, y: np.ndarray) -> list[np.ndarray]:
        predictions = self.predict(X)
        return [
            confusion_matrix(y[:, i], predictions[i])
            for i in range(self.n_classes)
        ]
