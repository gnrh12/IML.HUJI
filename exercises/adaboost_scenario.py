import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise),\
                                           generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ensemble = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    models_amounts = np.arange(1, n_learners + 1)

    train_errors = []
    test_errors = []
    for T in models_amounts:
        train_errors.append(ensemble.partial_loss(train_X, train_y, T))
        test_errors.append(ensemble.partial_loss(test_X, test_y, T))

    frame1 = go.Scatter(x=models_amounts, y=train_errors, name="Train Errors", mode="lines")
    frame2 = go.Scatter(x=models_amounts, y=test_errors, name="Test Errors", mode="lines")
    fig = go.Figure([frame1, frame2], layout=go.Layout(title={"text":f"Train and Test errors as a Function of Number of Fitted Models, Noise level - {noise}"}, xaxis={"title": "Number of Fitted Models"}, yaxis={"title": "Error Value"}))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T

    SYMBOLS = np.array(["circle", "x"])
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=[rf"$Ensemble-size: {{{t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(
            lambda X: ensemble.partial_predict(X, t), lims[0], lims[1],
            showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(
                                       color=np.where(test_y < 0, 0, 1),
                                       symbol=SYMBOLS[
                                           np.where(test_y < 0, 0, 1)],
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color="black", width=1)))],
                       rows=1, cols=(i % 4) + 1)
    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of Ensembles of Different Sizes, {noise} noise}}$",
        margin=dict(t=100)).update_xaxes(visible=False)\
        .update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    from IMLearn import metrics
    best_iterations_amount = np.argmin(test_errors) + 1
    best_accuracy = metrics.accuracy(
        test_y, ensemble.partial_predict(test_X, best_iterations_amount))

    T = [best_iterations_amount]
    fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.01,
                        vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(
            lambda X: ensemble.partial_predict(X, t), lims[0], lims[1],
            showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(
                           color=np.where(test_y < 0, 0, 1),
                           symbol=SYMBOLS[
                               np.where(test_y < 0, 0, 1)],
                           colorscale=[custom[0], custom[-1]],
                           line=dict(color="black", width=1)))],
            rows=1, cols=(i % 4) + 1)
    fig.update_layout(
        title=rf"$\textbf{{Decision Boundary Of Ensemble of Size {best_iterations_amount}, {noise} noise. Accuracy: {best_accuracy}}}$",
        margin=dict(t=100)).update_xaxes(visible=False) \
        .update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    T = [n_learners]
    fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.01,
                        vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(
            lambda X: ensemble.partial_predict(X, t), lims[0], lims[1],
            showscale=False),
            go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(
                           size=(ensemble.D_ / np.max(ensemble.D_) * 5),
                           color=np.where(train_y < 0, 0, 1),
                           symbol=SYMBOLS[np.where(train_y < 0, 0, 1)],
                           colorscale=[custom[0], custom[-1]],
                           line=dict(color="black", width=1)))],
                       rows=1, cols=i + 1)
    fig.update_layout(
        title=rf"$\textbf{{Decision Boundary of Ensemble of Size {n_learners}, Point Size Proportional to Weight}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
