import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    matrix = pd.read_csv(filename,
                         parse_dates=['Date']).dropna().drop_duplicates()

    for feature in ["Year", "Month", "Day"]:
        matrix = matrix[matrix[feature] > 0]

    matrix = matrix[matrix["Temp"] >= -72.777]

    days = [pd.Period(matrix["Date"][index], freq="D").day_of_year for index in
            matrix.index]
    matrix["DayOfYear"] = days

    return matrix


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    matrix = load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel = matrix.loc[matrix["Country"] == "Israel"]
    israel = israel.astype({"Year": str})

    px.scatter(israel, x=israel["DayOfYear"], y=israel["Temp"],
               labels={"DayOfYear": f"Day Of Year", "Temp": "Temperature"},
               color="Year",
               title="Temperature as a Function of Day of The Year in Israel")\
        .show()

    months = israel.groupby("Month").agg({"Temp": "std"}).reset_index()
    px.bar(months, x="Month", y="Temp", labels={"Month": f"Month of Year",
                                                "Temp": "Deviation"},
               title="Temperature Standard Deviation as a Function of Month of The Year in Israel")\
        .show()

    # Question 3 - Exploring differences between countries

    countries_months = matrix.groupby(["Month", "Country"]).agg({"Temp": "mean"}).reset_index()
    countries_months["err"] = np.std(countries_months["Temp"])/4
    px.line(countries_months, x="Month", y="Temp", error_y="err",
               labels={"Month": f"Month of Year", "Temp": "Temperature"},
               color="Country",
               title="Temperature as a Function of Month of The Year") \
        .show()

    # Question 4 - Fitting model for different values of `k`
    y = israel["Temp"]
    train_X, train_y, test_X, test_y = split_train_test(israel, y, 0.75)
    train_X, test_X = train_X['DayOfYear'], test_X['DayOfYear']
    loss_by_k = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k).fit(train_X[:], train_y[:])
        loss_by_k += [poly_model.loss(test_X[:], test_y[:])]

    for k in range(len(loss_by_k)):
        print("for degree " + str(k+1) + " loss is: " + str(loss_by_k[k]))

    px.bar(x=range(1, 11), y=loss_by_k, labels={"x": "Polynom Degree",
                                                "y": "Prediction Loss"},
           title="Temperature Prediction Loss as a Function of Polynom Degree")\
        .show()

    # Question 5 - Evaluating fitted model on different countries
    israel_X = israel["DayOfYear"]
    poly_model = PolynomialFitting(4).fit(israel_X, y)

    countries = matrix["Country"].loc[matrix["Country"] != "Israel"].unique()
    loses = []
    for country in countries:
        country = matrix.loc[matrix["Country"] == country]
        loses += [poly_model.loss(country["DayOfYear"], country["Temp"])]

    px.bar(x=countries, y=loses, labels={"x": "Country",
                                         "y": "Prediction Loss"},
           title="Prediction Loss of Israel-Trained Model as a Function of Country")\
        .show()
