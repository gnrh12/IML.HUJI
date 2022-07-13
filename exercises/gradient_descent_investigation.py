import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters

    """
    values, weights_list = [], []

    def callback(solver: GradientDescent, weights: np.ndarray, val: np.ndarray,
                 grad: np.ndarray, t: int, eta: float, delta: float):
        values.append(val)
        weights_list.append(weights)

    return callback, values, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module, name in [(L1, "L1"), (L2, "L2")]:
        losses = []
        for eta in etas:
            cur_module = module(init)
            callback, values, weights_list = get_gd_state_recorder_callback()
            gd_algorithm = GradientDescent(FixedLR(eta), callback=callback)
            res = gd_algorithm.fit(cur_module, None, None)
            plot_descent_path(module, np.array(weights_list),
                              f"- {name}, eta = {eta}").show()
            gd_iterations = np.array(range(len(values)))
            figure1 = go.Scatter(x=gd_iterations, y=values,
                                 mode="markers+lines")
            go.Figure(figure1, layout=go.Layout(
                title=f"{name} Convergence Rate- eta = {eta}",
                xaxis=dict(title="Gradient Descent Iteration"),
                yaxis=dict(title=f"{name} Norm Value"))
                      ).show()
            losses.append(np.min(values))
            print(f"lowest loss achieved for module {name}, eta {eta}- {losses[-1]}")
        print(f"lowest loss achieved for {name}- {np.min(losses)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    figure = go.Figure()
    losses = []
    for gamma in gammas:
        module = L1(init)
        callback, values, weights_list = get_gd_state_recorder_callback()
        gd_algorithm = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        res = gd_algorithm.fit(module, None, None)

        gd_iterations = np.array(range(len(values)))
        figure.add_trace(go.Scatter(x=gd_iterations, y=values,
                         mode="markers+lines", name=f"gamma = {gamma}"))

        losses.append(np.min(values))
        print(f"lowest loss achieved for L1 module, with gamma {gamma}- {losses[-1]}")
    print(f"lowest loss achieved for L1 module- {np.min(losses)}")


    # Plot algorithm's convergence for the different values of gamma
    figure.update_layout(
        title=f"L1 Module with Exponential LR Convergence Rates",
        xaxis=dict(title="Gradient Descent Iteration"),
        yaxis=dict(title="L1 Norm Value")).show()

    # Plot descent path for gamma=0.95
    module = L1(init)
    gamma = .95
    callback, values, weights_list = get_gd_state_recorder_callback()
    gd_algorithm = GradientDescent(ExponentialLR(eta, .95), callback=callback)
    res = gd_algorithm.fit(module, None, None)
    plot_descent_path(L1, np.array(weights_list),
                      f" Decent Path- L1 with gamma .95").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression(True,
                               GradientDescent(learning_rate=FixedLR(1e-4),
                                               max_iter=20000))
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(
        y_train, model.predict_proba(X_train.to_numpy()))

    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    alpha_star_index = np.argmax(tpr - fpr)
    alpha_star = thresholds[alpha_star_index]
    print(f"Alpha Star- {alpha_star}")
    from IMLearn.metrics.loss_functions import misclassification_error
    test_error \
        = misclassification_error(
        y_test.to_numpy(), model.predict_proba(X_test.to_numpy()) >= alpha_star)
    print(f"Test Error under Alpha Star- {test_error}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from IMLearn.model_selection import cross_validate
    for penalty in ["l1", "l2"]:
        best_lam, best_validation = 0, np.inf
        for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
            model = LogisticRegression(True,
                                       GradientDescent(FixedLR(1e-4),
                                                       max_iter=20000),
                                       penalty, lam)

            train, validation = cross_validate(model, X_train.to_numpy(),
                                               y_train.to_numpy(),
                                               misclassification_error)

            if validation < best_validation:
                best_validation, best_lam = validation, lam

        print(f"Best Lambda Value- {best_lam}, with Validation Value- {best_validation}")
        new_model = LogisticRegression(True,
                                       GradientDescent(FixedLR(1e-4),
                                                       max_iter=20000),
                                       penalty, best_lam)
        new_model.fit(X_train.to_numpy(), y_train.to_numpy())
        test_error = new_model.loss(X_test.to_numpy(), y_test.to_numpy())
        print(f"New Fitted Model with Lambda = {best_lam} Achieved Test Error = {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
