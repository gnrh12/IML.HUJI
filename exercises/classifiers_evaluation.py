from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"


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
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load("../datasets/" + f)
        samples = data[:, 0:2]
        lables = data[:, 2]

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        # callback = lambda perceptr, sample, res: losses.append([perceptr.loss(samples, lables)])

        perceptron = Perceptron(callback=lambda perceptr, sample, res:
                                losses.append(perceptr.loss(samples,
                                                             lables))).fit(samples, lables)
        print(losses)
        print(len(losses))

        # Plot figure
        # go.line(losses, x=range(1, 1001), y=losses,
        #            labels={"x": f"Training Iterations", "y": "Training Loss Values"},
        #            title="Training Loss Values as a Function of Training Iterations of a Perceptron Learner") \
        #     .show()

        # iterations = np.linspace(1, len(losses)).astype(int)
        go.Figure(
            [go.Scatter(x=list(range(len(losses))), y=np.array(losses), mode='lines',
                        name=r'$\text{training loss}$',
                        showlegend=True)],
            layout=go.Layout(
                title=r"Training Loss Values as a Function of Training Iterations of a Perceptron Learner, " + n + " data",
                xaxis_title=r"$\text{Training Iterations}$",
                yaxis_title=r"r$\text{Training Loss Values}$",
                height=300)).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples, lables = load_dataset(f)

        # Fit models and predict over training set
        lda_model = LDA().fit(samples, lables)
        lda_pred = lda_model.predict(samples)

        gaussNaive_model = GaussianNaiveBayes().fit(samples, lables)
        gauss_pred = gaussNaive_model.predict(samples)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        accuracies = [accuracy(lables, lda_pred), accuracy(lables, gauss_pred)]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Accuracy of LDA Model: ",
                                            "Accuracy of Gaussian Naive Bayes Model: "])

        lda_ellipses = [get_ellip]


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
