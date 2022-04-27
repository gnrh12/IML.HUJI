from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import loss_functions

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        self.vars_ = np.zeros(self.classes_.size, X.shape[1])
        for k in range(self.classes_.size):
            for i in range(X.shape[0]):
                if y[i] == self.classes_[k]:
                    mu = self.mu_[k]
                    self.vars_[k] += np.square(X[i] - mu) / (counts[k] - 1)

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
        likelihoods = self.likelihood(X)
        for i in range(X.shape[0]):
            results.append(self.classes_[np.argmax(likelihoods[i])])

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
        for i in range(X.shape[0]):
            for k in self.classes_.size:
                cur_likli = self.pi_[k]
                for d in range(X.shape[1]):
                    Z = np.sqrt(np.power(2 * np.pi, X.shape[1])
                                * self.vars_[k][d])
                    diff = np.power(X[i][d] - self.mu_[k][d], 2)
                    exp = np.exp(-0.5 * diff / self.vars_[k][d])
                    cur_likli *= (1 / Z) * exp
                likelihoods[i, k] = cur_likli

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
