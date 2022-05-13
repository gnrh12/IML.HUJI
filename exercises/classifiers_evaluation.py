from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        samples, lables = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

<<<<<<< HEAD
        perceptron = Perceptron(callback=lambda perceptr, sample, res:
                                losses.append(perceptr.loss(samples,
                                                             lables)))\
            .fit(samples, lables)

        # Plot figure of loss as function of fitting iteration
        go.Figure(
            [go.Scatter(x=list(range(len(losses))), y=np.array(losses),
                        mode='lines', name=r'$\text{training loss}$',
                        showlegend=True)],
            layout=go.Layout(
                title=r"Training Loss Values as a Function of Training"
                      r" Iterations of a Perceptron Learner, " + n + " data",
                xaxis_title=r"$\text{Training Iterations}$",
                yaxis_title=r"r$\text{Training Loss Values}$",
                height=300)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")
=======
        # Plot figure of loss as function of fitting iteration
        raise NotImplementedError()
>>>>>>> 072d4a39a901a32c752ab820cf8b7ec30a77b344


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples, labels = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda_model = LDA().fit(samples, labels)
        lda_pred = lda_model.predict(samples)

        gaussNaive_model = GaussianNaiveBayes().fit(samples, labels)
        gauss_pred = gaussNaive_model.predict(samples)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
<<<<<<< HEAD
        lda_acc = accuracy(labels, lda_pred)
        gauss_acc = accuracy(labels, gauss_pred)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles
                            =["Accuracy of LDA Model: " + str(lda_acc),
                                            "Accuracy of Gaussian"
                                            " Naive Bayes Model: "
                              + str(gauss_acc)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        fig.update_layout(title="Decision Boundaries by Model, " + f
                                + " Dataset", margin=dict(t=100,))\
            .update_xaxes(visible=False)\
            .update_yaxes(visible=False)

        # Add traces for data-points setting symbols and colors
        lims = np.array([samples.min(axis=0, initial=0),
                            samples.max(axis=0, initial=0)]).T \
            + np.array([-.4, .4])
        fig.add_traces([decision_surface(lda_model.predict, lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(
                                       color=lda_model.predict(samples),
                                               symbol=class_symbols[labels],
                                               colorscale=class_colors(3),
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=1, cols=1)
        fig.add_traces([decision_surface(gaussNaive_model.predict, lims[0],
                                         lims[1], showscale=False),
                        go.Scatter(x=samples[:, 0], y=samples[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(
                                       color=gaussNaive_model.predict(samples),
                                               symbol=class_symbols[labels],
                                               colorscale=class_colors(3),
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=1, cols=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=[lda_model.mu_[k][0]],
                                   y=[lda_model.mu_[k][1]],
                                   mode="markers", showlegend=False,
                                   marker=dict(color="black", symbol="x",
                                   line=dict(color="black", width=4)))
                        for k in range(lda_model.classes_.size)],
                       rows=1, cols=1)

        fig.add_traces([go.Scatter(x=[gaussNaive_model.mu_[k][0]],
                                   y=[gaussNaive_model.mu_[k][1]],
                                   mode="markers", showlegend=False,
                                   marker=dict(color="black", symbol="x",
                                   line=dict(color="black", width=4)))
                        for k in range(gaussNaive_model.classes_.size)],
                       rows=1, cols=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        lda_ellipses = [get_ellipse(lda_model.mu_[k], lda_model.cov_)
                        for k in range(lda_model.classes_.size)]
        gaussNaive_ellipses = [get_ellipse(gaussNaive_model.mu_[k],
                                           np.diag(gaussNaive_model.vars_[k]))
                               for k in range(gaussNaive_model.classes_.size)]

        fig.add_traces(lda_ellipses, rows=1, cols=1)
        fig.add_traces(gaussNaive_ellipses, rows=1, cols=2)

        fig.show()
=======
        raise NotImplementedError()

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()

>>>>>>> 072d4a39a901a32c752ab820cf8b7ec30a77b344

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
