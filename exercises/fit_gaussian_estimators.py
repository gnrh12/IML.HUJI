from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    uni_gaussian = UnivariateGaussian().fit(samples)
    print(uni_gaussian.mu_, uni_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    samples_sizes = np.linspace(10, 1000, 100).astype(int)
    samples_by_amount = [samples[:i] for i in samples_sizes]
    models = [UnivariateGaussian().fit(j) for j in samples_by_amount]
    abs_diffs = [np.abs(model.mu_ - mu) for model in models]
    go.Figure([go.Scatter(x=samples_sizes, y=abs_diffs, mode='lines+markers',
                          name=r'$\text{absolute difference}$',
                          showlegend=True)],
              layout=go.Layout(
                  title=r"$\text{(2) Distance Between Estimated And True"
                        r" Expectations As a Function Of Sample Size}$",
                              xaxis_title=r"$\text{number of samples}$",
                              yaxis_title=r"r$\text{absolute difference}$",
                              height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=samples, y=uni_gaussian.pdf(samples),
                          mode='markers', name=r'$\text{PDF values}$',
                          showlegend=True)],
              layout=go.Layout(
                  title=r"$\text{(3) PDF Values Of Drawn Samples}$",
                  xaxis_title=r"$\text{sample value}$",
                  yaxis_title=r"r$\text{PDF value}$",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)
    multi_gaussian = MultivariateGaussian().fit(samples)
    print(multi_gaussian.mu_)
    print(multi_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f_space = np.linspace(-10, 10, 200)
    likelihood_vals = []
    for a in f_space:
        likelihood_vals.append([multi_gaussian
                               .log_likelihood(np.array([a, 0, b, 0]),
                                               cov, samples) for b in f_space])

    go.Figure(go.Heatmap(x=f_space, y=f_space, z=likelihood_vals),
              layout=go.Layout(
                  title=r"$\text{(5) Log-Likelihood Values As A Function Of"
                        r" Expectation}$",
                  xaxis_title=r"$\text{f1 values}$",
                  yaxis_title=r"r$\text{f3 values}$",
                  height=500,
                  width=500)).show()

    # Question 6 - Maximum likelihood
    print("Max log-likelihood is achieved by f1 = 3.969, f3 = -0.050")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()