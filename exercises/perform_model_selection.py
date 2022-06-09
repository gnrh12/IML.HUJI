from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # diff = 2 + 1.2
    # X = (np.random.rand(n_samples) * diff) - 1.2 # if samples should be drawn uniformly

    X = np.linspace(-1.2, 2, n_samples)
    true_y = (X + 3)*(X + 2)*(X + 1)*(X - 1)*(X - 2)

    indices = np.argsort(X)
    X = X[indices]
    true_y = true_y[indices]
    noise_y = true_y + np.random.randn(n_samples) * np.sqrt(noise)

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(noise_y),
                                                        2/3)
    train_X, train_y, test_X, test_y = train_X.to_numpy().flatten(),\
                                       train_y.to_numpy().flatten(),\
                                       test_X.to_numpy().flatten(),\
                                       test_y.to_numpy().flatten()


    frame1 = go.Scatter(x=X, y=true_y, name="Noiseless Model", mode='lines')
    frame2 = go.Scatter(x=train_X, y=train_y, name="Train Set", mode="markers")
    frame3 = go.Scatter(x=test_X, y=test_y, name="Test Set", mode="markers")
    fig = go.Figure([frame1, frame2, frame3], layout=go.Layout(title={
        "text": f"Dataset Plot- Noiseless set, Train set and Test set. Noise Level- {noise}, Amount of Samples- {n_samples}"},
           xaxis={"title": "X"},
           yaxis={"title": "y"}))
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    degrees = np.arange(11)
    train_errors = []
    validation_errors = []

    for k in range(11):
        train_error, validation_error = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error)
        train_errors.append(train_error)
        validation_errors.append(validation_error)

    frame1 = go.Scatter(x=degrees, y=train_errors, name="Train Errors", mode="lines")
    frame2 = go.Scatter(x=degrees, y=validation_errors, name="Validation Errors", mode="lines")
    fig = go.Figure([frame1, frame2], layout=go.Layout(title={
        "text": f"Test and Validation Errors as a Function of Polynomial Degree. Noise- {noise}, Amount of Samples- {n_samples}"},
        xaxis={"title": "Polynomial Degree"},
        yaxis={"title": "Error"}))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_errors)
    model = PolynomialFitting(best_k).fit(train_X, train_y)
    error = mean_square_error(test_y, model.predict(test_X))
    print("Q3: value of best k: " + str(best_k))
    print("Q3: test error achieved with " + str(best_k) + ": " + str(error))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_samples,
                                                        shuffle=False)
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 2, n_evaluations)

    ridge_train_errors = []
    ridge_validation_errors = []
    lasso_train_errors = []
    lasso_validation_errors = []

    for l in lambdas:
        ridge_train_error, ridge_validation_error \
            = cross_validate(RidgeRegression(l),
                             X_train, y_train, mean_square_error)
        lasso_train_error, lasso_validation_error \
            = cross_validate(Lasso(l), X_train, y_train, mean_square_error)

        ridge_train_errors.append(ridge_train_error)
        ridge_validation_errors.append(ridge_validation_error)
        lasso_train_errors.append(lasso_train_error)
        lasso_validation_errors.append(lasso_validation_error)

    frame1 = go.Scatter(x=lambdas, y=ridge_train_errors, name="Ridge Train Errors", mode="lines")
    frame2 = go.Scatter(x=lambdas, y=ridge_validation_errors, name="Ridge Validation Errors", mode="lines")
    frame3 = go.Scatter(x=lambdas, y=lasso_train_errors, name="Lasso Train Errors", mode="lines")
    frame4 = go.Scatter(x=lambdas, y=lasso_validation_errors, name="Lasso Validation Errors", mode="lines")
    fig = go.Figure([frame1, frame2, frame3, frame4], layout=go.Layout(title={
        "text": f"Test and Validation Errors as a Function of Regularization "
                f"Parameter, Ridge and Lasso Regularizations"},
        xaxis={"title": "Regularization Parameter"},
        yaxis={"title": "Error"}))
    fig.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_parameter = lambdas[np.argmin(ridge_validation_errors)]
    best_lasso_parameter = lambdas[np.argmin(lasso_validation_errors)]
    ridge = RidgeRegression(best_ridge_parameter).fit(X_train, y_train)
    lasso = Lasso(best_lasso_parameter).fit(X_train, y_train)
    ls = LinearRegression().fit(X_train, y_train)

    ridge_error = mean_square_error(y_test, ridge.predict(X_test))
    lasso_error = mean_square_error(y_test, lasso.predict(X_test))
    ls_error = mean_square_error(y_test, ls.predict(X_test))
    print("\nQ8:")
    print("value of best ridge parameter: " + str(best_ridge_parameter))
    print("value of best lasso paramater: " + str(best_lasso_parameter))
    print("test error achieved with ridge and paramater = " + str(best_ridge_parameter) + ": " + str(ridge_error))
    print("test error achieved with lasso and paramater = " + str(best_lasso_parameter) + ": " + str(lasso_error))
    print("test error achieved with ls: " + str(ls_error))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()

