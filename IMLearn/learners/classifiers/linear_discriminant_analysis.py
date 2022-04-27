from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)

        self.mu_ = np.zeros(self.classes_.size, X.shape[1])
        for i in range(self.classes_.size):
            self.mu_[i] = np.sum(
                [x_i if (y[i] == self.classes_[i]) else 0 for x_i in X])\
                          / counts[i]

        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        sum = 0
        for i in range(X.shape[0]):
            y_index = np.where(self.classes_ == y[i])
            diff = X[i] - self.mu_[y_index]
            sum += diff.reshape(diff.size, 1) @ diff.reshape(1, diff.size)
        self.cov_ = sum / (X.shape[1] - self.classes_.size)

        self._cov_inv = inv(self.cov_)

        self.pi_ = np.array([counts[i] / X.shape[0] for i in range(self.classes_.shape[0])])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        results = []
        for x in X:
            ks_array = []
            for k in self.classes_.size:
                ak = self._cov_inv @ self.mu_[k].reshape(self.mu_[k].size, 1)
                bk = np.log(self.pi_[k]) \
                    - ((self.mu_[k].reshape(1, self.mu_[k].size)
                        @ self._cov_inv
                        @ self.mu_[k].reshape(self.mu_[k], 1)) / 2)
                ks_array.append(ak.reshape(1, ak.size)
                                @ x.reshape(x.size, 1) + bk)
            results.append(self.classes_[np.argmax(ks_array)])

        return np.array(results)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros(X.shape[0], self.classes_.size)
        Z = np.sqrt(np.power(2 * np.pi, X.shape[1]) * det(self.cov_))
        for i in range(X.shape[0]):
            for k in self.classes_.size:
                diff = X[i] - self.mu_[k]
                exp = np.exp(-(0.5 * diff.reshape(1, diff.size)
                               @ self._cov_inv @ diff.reshape(diff.size, 1)))
                likelihoods[i, k] = self.pi_[k] * (1 / Z) * exp

        return likelihoods


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return loss_functions.misclassification_error(y, self.predict(X))
