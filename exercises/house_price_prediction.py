from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    matrix = pd.read_csv(filename).dropna().drop_duplicates()

    matrix["zipcode"] = matrix["zipcode"].astype(int)

    for feature in ["id", "date", "lat", "long"]:
        matrix.drop(feature, inplace=True, axis=1)

    for feature in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built",
              "sqft_living15", "sqft_lot15"]:
        matrix = matrix[matrix[feature] > 0]

    for feature in ["bedrooms", "bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        matrix = matrix[matrix[feature] >= 0]

    matrix = matrix[matrix["waterfront"].isin([0, 1]) &
            matrix["view"].isin(range(5)) &
            matrix["condition"].isin(range(1, 6)) &
            matrix["grade"].isin(range(1, 14))]

    # matrix["decade_built"] = (matrix["yr_built"] // 100).astype(int)
    # matrix.drop("yr_built", 1)

    matrix = pd.get_dummies(matrix, columns=['zipcode'])

    matrix = matrix[matrix["bedrooms"] < 20]
    matrix = matrix[matrix["sqft_living"] < 15000]
    matrix = matrix[matrix["sqft_living15"] < 15000]
    matrix = matrix[matrix["sqft_lot"] < 150000]
    matrix = matrix[matrix["sqft_lot15"] < 150000]

    return matrix.drop("price", 1), matrix.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X:
        if feature.startswith("zipcode") or feature.startswith("yr_built"):
            continue
        pearson_correlation = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
            # X[feature].to_frame().corrwith(y)
        px.scatter(pd.DataFrame({'x': X[feature], 'y': y}),
                   x="x", y="y",
                   title=f"Pearson Correlation Between {feature} Feature and Response. Correlation: {pearson_correlation}",
                   labels={"x": f"{feature} Values",
                           "y": "Response Vector Values"},
                   width=1000) \
            .write_image(output_path + "/" + "pearson_correlation_%s.png" % feature)


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = (load_data("datasets/house_prices.csv"))

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "pearson_correlations")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    averages = []
    variances = []
    percentages = np.arange(10, 101, 1)
    for p in percentages:
        p_loss = []
        for j in range(10):
            cur_X = train_X.sample(frac=p/100)
            cur_y = train_y.reindex_like(cur_X)
            model = LinearRegression(True)
            model.fit(cur_X[:], cur_y[:])
            p_loss += [model.loss(test_X[:], test_y[:])]
        p_loss = np.array(p_loss)
        averages += [p_loss.mean()]
        variances += [p_loss.std()]
    averages = np.array(averages)
    variances = np.array(variances)


    frame1 = go.Scatter(x=percentages, y=averages, name="Loss Values", mode="markers+lines", )
    frame2 = go.Scatter(x=percentages, y=averages-2*variances, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False)
    frame3 = go.Scatter(x=percentages, y=averages+2*variances, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)
    fig = go.Figure([frame1, frame2, frame3], layout=go.Layout(title={"text":"MSE Loss of House Price Prediction as a Function of Training Data Percentage Used"}, xaxis={"title": "Training Data Percentage Used"}, yaxis={"title": "Prediction MSE Loss"}))
    fig.show()
