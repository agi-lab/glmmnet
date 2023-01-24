"""
Some helper functions for the demo notebooks.
"""
import os
import logging
import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh.plotting as bop
import bokeh.models as bom

from bokeh.layouts import gridplot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import norm, gamma, lognorm
from scipy.special import beta

def split(X, y, n_train):
    """
    Deterministic split of the data into training and test sets at n_train.
    """
    X_train = X.iloc[:n_train, :]
    X_test = X.iloc[n_train:, :]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

def crps_norm(y, loc, scale):
    """
    Compute CRPS of a location-scale transformed normal distribution.
    Translated from the R package `scoringRules`.
    Source code: https://github.com/FK83/scoringRules/blob/master/R/scores_norm.R
    """
    y = np.array(y, dtype=float)
    y = y - loc
    z = np.divide(y, scale, out=np.zeros_like(y), where=(~ np.isclose(y, 0) | ~ np.isclose(scale, 0)))
    crps = scale * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return crps

def crps_gamma(y, shape, scale):
    """
    Compute CRPS of a gamma distribution.
    Translated from the R package `scoringRules`.
    Source code: https://github.com/FK83/scoringRules/blob/master/R/scores_gamma.R
    """
    y = np.array(y, dtype=float)
    p1 = gamma.cdf(y, a=shape, scale=scale)
    p2 = gamma.cdf(y, a=np.add(shape, 1), scale=scale)
    crps = y * (2*p1 - 1) - scale * (shape * (2*p2 - 1) + 1 / beta(0.5, shape))
    return crps

def crps_lognorm(y, meanlog, sdlog):
    """
    Compute CRPS of a lognormal distribution.
    Translated from the R package `scoringRules`.
    Source code: https://github.com/FK83/scoringRules/blob/master/R/scores_lnorm.R
    """
    y = np.array(y, dtype=float)
    c1 = y * (2 * lognorm.cdf(y, s=sdlog, scale=np.exp(meanlog)) - 1)
    c2 = 2 * np.exp(np.add(meanlog, np.power(sdlog, 2) / 2))
    c3 = lognorm.cdf(y, s=sdlog, scale=np.exp(np.add(meanlog, np.power(sdlog, 2)))) + norm.cdf(sdlog / np.sqrt(2)) - 1
    crps = c1 - c2 * c3
    return crps

def evaluate_predictions(y, y_pred, categories, likelihood="gaussian", **kwargs):
    """
    Evaluate the performance of a model based on the predictions.
    Specify the likelihood of the predictions with the keyword argument `likelihood`.

    Parameters
    ----------
    y: true values
    y_pred: predicted values
    categories: categories of the observations
    likelihood: form of the response distribution, one of the following:
        - "gaussian": normal distribution;
        - "gamma": gamma distribution;
        - "lognorm": lognormal distribution;
        - "loggamma": loggamma distribution.
    **kwargs: additional keyword arguments to specify the estimated parameters of the likelihood (to compute CRPS):
        - "gaussian" requires `loc` and `scale`;
        - "gamma" requires `shape` (or synonymously `gamma_shape`) and `gamma_scale`;
        - "lognorm" requires `meanlog` and `sdlog`.
        - "loggamma" requires `shape` (or synonymously `gamma_shape`) and `gamma_scale`.
    """
    if likelihood == "gaussian":
        loc = kwargs.get("loc", y_pred)
        scale = kwargs.get("scale", np.sqrt(mean_squared_error(y, y_pred)))

    # Take y and y_pred back to the original scale
    if likelihood == "lognorm":
        # Transform the observations back to the original scale
        y = np.exp(y)
        # Transform the predictions back to the original scale
        meanlog = kwargs.get("meanlog", y_pred)                                      # Infer the predictive parameters from predictions
        sdlog = kwargs.get("sdlog", np.sqrt(mean_squared_error(np.log(y), meanlog))) # Infer the predictive parameters from predictions
        y_pred = np.exp(np.add(meanlog, np.power(sdlog, 2) / 2))

    if likelihood in ["gamma", "loggamma"]:
        gamma_shape = kwargs.get("gamma_shape", kwargs.get("shape", None))
        gamma_scale = kwargs.get("gamma_scale", None)
        if gamma_shape is None:
            # Estimate shape parameter of gamma distribution by Pearson's method
            # Related discussion: https://stats.stackexchange.com/questions/367560/
            # `statsmodels` ref: https://www.statsmodels.org/dev/_modules/statsmodels/genmod/generalized_linear_model.html#GLM.estimate_scale
            resid = np.power(y - y_pred, 2)
            var = np.power(y_pred, 2)
            gamma_dispersion = np.sum(resid / var) / len(y)
            gamma_shape = 1 / gamma_dispersion
        if gamma_scale is None:
            gamma_scale = y_pred / gamma_shape

        if likelihood == "loggamma":
            if any(gamma_scale >= 1):
                raise ValueError("The scale parameter of the loggamma distribution must be < 1 otherwise expectation is infinite.")
            # Transform the observations back to the original scale
            y = np.exp(y)
            # Transform the predictions back to the original scale
            y_pred = np.exp(y_pred)

    scores = dict()

    scores["MAE"] = mean_absolute_error(y, y_pred)
    scores["MedAE"] = median_absolute_error(y, y_pred) # for a more robust estimate of the error
    scores["MedPE"] = np.median(np.divide(np.abs(np.subtract(y, y_pred)), y)) # median percentage error
    scores["RMSE"] = np.sqrt(mean_squared_error(y, y_pred))
    scores["R2"] = r2_score(y, y_pred)

    # RMSE of average prediction for each category
    data = pd.concat([
        pd.Series(categories, name="category").reset_index(drop=True), 
        pd.Series(y, name="y").reset_index(drop=True), 
        pd.Series(y_pred, name="y_pred").reset_index(drop=True)
        ], axis=1)
    gb = data.groupby("category", as_index=False)
    counts = gb.size()
    avg_by_cat = gb[["y", "y_pred"]].mean()
    scores["RMSE_avg"] = np.sqrt(mean_squared_error(avg_by_cat["y"], avg_by_cat["y_pred"]))

    # Volume weighted RMSE of average prediction for each category
    scores["RMSE_avg_weighted"] = np.sqrt(
        mean_squared_error(avg_by_cat["y"], avg_by_cat["y_pred"], sample_weight=counts["size"]))

    # CRPS to quantify accuracy of probabilistic predictions
    if likelihood == "gaussian":
        scores["CRPS"] = crps_norm(y, loc, scale).mean()
    elif likelihood == "gamma":
        scores["CRPS"] = crps_gamma(y, gamma_shape, gamma_scale).mean()
    elif likelihood == "lognorm":
        scores["CRPS"] = crps_norm(np.log(y), meanlog, sdlog).mean()
    elif likelihood == "loggamma":
        scores["CRPS"] = crps_gamma(np.log(y), gamma_shape, gamma_scale).mean()
    
    # Negative log-likelihood of probabilistic predictions
    if likelihood == "gaussian":
        scores["NLL"] = -norm.logpdf(y, loc=loc, scale=scale).mean()
    elif likelihood == "gamma":
        scores["NLL"] = -gamma.logpdf(y, a=gamma_shape, scale=gamma_scale).mean()
    elif likelihood == "lognorm":
        scores["NLL"] = -np.log(norm.pdf(np.log(y), loc=meanlog, scale=sdlog) / y).mean()
    elif likelihood == "loggamma":
        scores["NLL"] = -np.log(gamma.pdf(np.log(y), a=gamma_shape, scale=gamma_scale) / y).mean()

    return scores

def evaluate_model(regressor, X, y, categories, likelihood="gaussian", **kwargs):
    """
    Evaluate a model on a dataset.
    """
    y_pred = regressor.predict(X)
    if (y_pred.ndim == 2) and (y_pred.shape[1] == 1):
        y_pred = y_pred.flatten()
    return evaluate_predictions(y, y_pred, categories, likelihood, **kwargs)

# Colour scheme: https://coolors.co/palette/231942-5e548e-9f86c0-be95c4-e0b1cb
def plot_from_model(model, X_train, y_train, X_test, y_test, log_scale=False, show=True):
    """
    Plot the predictions vs. the true values for both the training and test sets.
    """
    if (log_scale):
        y_axis_type = "log"
        x_axis_type = "log"
    else:
        y_axis_type = "linear"
        x_axis_type = "linear"

    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    y_train, y_test = pd.Series(y_train), pd.Series(y_test)
    target = pd.concat([y_train, y_test])
    left, right = target.min(), target.max()
    span = right - left
    bottom, top = target.min() - span * 0.1, target.max() + span * 0.1
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    if (y_pred_train.ndim == 2) and (y_pred_train.shape[1] == 1):
        y_pred_train = y_pred_train.flatten()
        y_pred_test = y_pred_test.flatten()
    p1 = bop.figure(
        title="Predictions vs ground truth", 
        x_axis_label="Ground truth", y_axis_label="Predictions", 
        x_range=(left, right), y_range=(bottom, top),
        x_axis_type=x_axis_type, y_axis_type=y_axis_type
    )
    p1.circle(y_train, y_pred_train, legend_label="In-sample", color="#5e548e")
    p1.legend.location = "top_left"
    p1.add_layout(bom.Slope(gradient=1, y_intercept=0, line_color="black", line_width=2))
    p1.output_backend = "svg"

    p2 = bop.figure(
        title="Predictions vs ground truth", 
        x_axis_label="Ground truth", y_axis_label="Predictions", 
        x_range=(left, right), y_range=(bottom, top),
        x_axis_type=x_axis_type, y_axis_type=y_axis_type
    )
    p2.circle(y_test, y_pred_test, legend_label="Out-of-sample", color="#9f86c0")
    p2.legend.location = "top_left"
    p2.add_layout(bom.Slope(gradient=1, y_intercept=0, line_color="black", line_width=2))
    p2.output_backend = "svg"

    grid = gridplot([[p1, p2]], width=500, height=400)
    if show:
        bop.show(grid)
    
    return grid

def plot_from_predictions(y_pred_train, y_train, y_pred_test, y_test, log_scale=False, show=True):
    """
    Plot the predictions vs. the true values for both the training and test sets.
    """
    if (log_scale):
        y_axis_type = "log"
        x_axis_type = "log"
    else:
        y_axis_type = "linear"
        x_axis_type = "linear"

    y_pred_train, y_pred_test = pd.Series(y_pred_train), pd.Series(y_pred_test)
    y_train, y_test = pd.Series(y_train), pd.Series(y_test)
    target = pd.concat([y_train, y_test])
    left, right = target.min(), target.max()
    span = right - left
    bottom, top = target.min() - span * 0.1, target.max() + span * 0.1

    p1 = bop.figure(
        title="Predictions vs ground truth", 
        x_axis_label="Ground truth", y_axis_label="Predictions", 
        x_range=(left, right), y_range=(bottom, top),
        x_axis_type=x_axis_type, y_axis_type=y_axis_type
    )
    p1.circle(y_train, y_pred_train, legend_label="In-sample", color="#5e548e")
    p1.legend.location = "top_left"
    p1.add_layout(bom.Slope(gradient=1, y_intercept=0, line_color="black", line_width=2))
    p1.output_backend = "svg"

    p2 = bop.figure(
        title="Predictions vs ground truth", 
        x_axis_label="Ground truth", y_axis_label="Predictions", 
        x_range=(left, right), y_range=(bottom, top),
        x_axis_type=x_axis_type, y_axis_type=y_axis_type
    )
    p2.circle(y_test, y_pred_test, legend_label="Out-of-sample", color="#9f86c0")
    p2.legend.location = "top_left"
    p2.add_layout(bom.Slope(gradient=1, y_intercept=0, line_color="black", line_width=2))
    p2.output_backend = "svg"

    grid = gridplot([[p1, p2]], width=500, height=400)
    if show:
        bop.show(grid)

    return grid

# Helper function to make tensorflow less verbose
def set_tf_loglevel(level):
    """
    Set the log level of TensorFlow.
    Source: https://stackoverflow.com/a/57439591
    """
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

# Helper function to plot loss curves of NN
def plot_loss_curves(history, show=True):
    """
    Plot the loss curves for the training and validation sets.
    """
    p = bop.figure(width=1000, height=400, title="Loss curves", x_axis_label="Epoch", y_axis_label="Loss")
    # Omit the first data point by taking [1:] for a better plot
    epochs = np.add(1, range(len(history["loss"]))[1:])
    p.line(epochs, history["loss"][1:], legend_label="Training", color="dodgerblue")
    p.line(epochs, history["val_loss"][1:], legend_label="Validation", color="lightskyblue")
    p.legend.location = "top_right"
    if show:
        bop.show(p)
    
    return p

def embedding_preproc(X_train, X_test, cat_cols):
    """
    Change data to list for processing with entity embeddings.
    Code adapted from https://github.com/oegedijk/keras-embeddings/blob/master/build_embeddings.py
    """
    input_list_train = []
    input_list_test = []

    for c in cat_cols:
        input_list_train.append(X_train[c].values)
        input_list_test.append(X_test[c].values)

    return input_list_train, input_list_test

def plot_ridgeline(categories, y, selected_cats, ax, title=None, ylabel=True, p_min=None, p_max=None):
    """
    Plot a ridgeline plot of the data for selected categories.
    Code adapted from https://scipython.com/blog/ridgeline-plots-of-monthly-uk-temperatures/
    """
    num_cats = len(selected_cats)
    if p_min is None or p_max is None:
        p_min, p_max = np.min(y), np.max(y)
    x_grid = np.linspace(p_min - 0.5, p_max + 0.5, 100)
    cmap = mpl.cm.get_cmap("viridis")
    offset = 0.25
    data = pd.DataFrame({"y": y, "category": categories})
    y_mean = data.groupby("category").mean()["y"]
    norm = mpl.colors.Normalize(vmin=y_mean.min(), vmax=y_mean.max())

    ax.yaxis.set_tick_params(length=0, width=0)
    ax.set_ylim(-0.01, num_cats * offset + 0.05)
    if ylabel:
        ax.set_yticks(np.arange(0, num_cats * offset, offset))
        ax.set_yticklabels([f"Category {i}" for i in selected_cats])
    else:
        ax.set_yticks([])

    for i, cat in enumerate(selected_cats):
        c = cmap(norm(y_mean[cat]))
        y_i = y[categories == cat]
        dist = scipy.stats.gaussian_kde(y_i)
        ax.plot(x_grid, dist(x_grid) / dist(x_grid).max() * 0.3 + offset * i, color="w", zorder=num_cats + 1 - i)
        ax.fill_between(x_grid, dist(x_grid) / dist(x_grid).max() * 0.3 + offset * i, offset * i, color=c, zorder=num_cats + 1 - i)
        ax.axhline(offset * i, color=c, zorder=num_cats + 1 - i, linewidth=1)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_title(title)

# Export the predictions to a csv file (for plotting in R)
def get_predictions(log_y, log_y_pred, dist, **kwargs):
    if dist == "lognormal":
        meanlog = log_y_pred
        sdlog = kwargs.get("sdlog", np.sqrt(mean_squared_error(log_y, log_y_pred)))
        return pd.DataFrame({
            "log_y": log_y,
            "meanlog": meanlog,
            "sdlog": sdlog,
        })
    elif dist == "loggamma":
        gamma_shape = kwargs.get("gamma_shape", None)
        if gamma_shape is None:
            # Estimate shape parameter of gamma distribution by Pearson's method
            # Related discussion: https://stats.stackexchange.com/questions/367560/
            # `statsmodels` ref: https://www.statsmodels.org/dev/_modules/statsmodels/genmod/generalized_linear_model.html#GLM.estimate_scale
            resid = np.power(log_y - log_y_pred, 2)
            var = np.power(log_y_pred, 2)
            gamma_dispersion = np.sum(resid / var) / len(log_y)
            gamma_shape = 1 / gamma_dispersion
        gamma_scale = log_y_pred / gamma_shape
        return pd.DataFrame({
            "log_y": log_y,
            "gamma_shape": gamma_shape,
            "gamma_scale": gamma_scale,
        })
