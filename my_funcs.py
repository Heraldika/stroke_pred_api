import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from typing import Callable

########################
# Annotation for plots #
########################


def vert_annot(axes: plt.axes, precision: int = 0) -> None:
    """
    Annotate vertical bars in a barplot with its value
    """
    for bar in axes.patches:
        axes.annotate(
            format(bar.get_height(), f".{precision}f"),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )


def hori_annot(axes: plt.axes, precision: int = 0) -> None:
    """
    Annotate vertical bars in a barplot with its value
    """
    for bar in axes.patches:
        axes.annotate(
            format(bar.get_width(), f".{precision}f"),
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
        )


def vert_text(ax: plt.axes, text: str, x: float, color: str = "k") -> None:
    """
    Vertical text annotation
    """
    y = ax.get_ylim()[1] * 0.5
    ax.annotate(
        text,
        (x, y),
        color=color,
        rotation="90",
        va="center",
        xytext=(4, 0),
        textcoords="offset points",
    )


#####################
# Ploting functions #
#####################


def plot_hist(df: pd.Series, ax: plt.axes, title: str = "") -> None:
    sns.histplot(df, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(ax.get_xlabel().capitalize())


def plot_qq(df: pd.Series, ax: plt.axes, title: str = "") -> None:
    sm.qqplot(df, line="45", fit=True, ax=ax)
    ax.set_title(title)


def plot_residuals(
    values: np.ndarray, errors: np.ndarray, ax: plt.axes, xlabel: str, title: str
) -> None:
    """
    Plots true values agiants residuals from linear regression
    """
    axi = sns.scatterplot(x=values, y=errors, alpha=0.5, ax=ax)
    axi.set_xlabel(xlabel)
    axi.set_ylabel("Residuals")
    axi.set_title(title, fontsize=15)
    axi.axhline(y=0, color="r", linestyle="-")
    sns.despine()


def plot_compare_residuals(
    fitted: np.ndarray,
    errors_fit: np.ndarray,
    prediction: np.ndarray,
    errors_pred: np.ndarray,
) -> None:
    """
    Plots fitted and predicted values agiants residuals from linear regression
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ax1 = plot_residuals(
        fitted, errors_fit, ax[0], "Fitted values", "Fitted values againts residuals"
    )

    ax2 = plot_residuals(
        prediction,
        errors_pred,
        ax[1],
        "Predicted values",
        "Predicted values againts residuals",
    )


def plot_reg(
    prediction: np.ndarray, data_y: np.ndarray, ax, ylabel: str, title: str
) -> None:
    """
    Plots a scatterplot for true y values and a linplot for predictions
    """
    x_points = range(len(data_y))
    ax = sns.scatterplot(
        x=x_points, y=data_y, color="b", alpha=0.5, label="True value", ax=ax
    )
    ax = sns.scatterplot(
        x=x_points, y=prediction, alpha=0.7, label="Prediction", ax=ax, color="#FF6767"
    )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Data points")
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    sns.despine()
    ax.legend(loc="upper right", bbox_to_anchor=(0.7, 0.95, 0.3, 0.3))


def plot_compare_reg(
    fitted: np.ndarray,
    fdata_y: np.ndarray,
    prediction: np.ndarray,
    pdata_y: np.ndarray,
    ylabel: str,
) -> None:
    """
    Plots predicted and fitted regression plots
    """
    fig, ax = plt.subplots(2, 1, figsize=(20, 6), sharey=True)
    fig.subplots_adjust(hspace=0.3)
    plot_reg(fitted, fdata_y, ax[0], ylabel, "Model fit vs actual values")
    plot_reg(prediction, pdata_y, ax[1], ylabel, "Predictions vs actual values")


def plot_ci(dist: np.ndarray, lb: float, mean_diff: float, ub: float) -> None:
    """
    Plots statistics distribution with upper and lower bounds.
    Takes in values from boot_mean_diff_ci
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.kdeplot(dist)

    ax.axvline(lb, color="b")
    ax.axvline(ub, color="b")
    ax.axvline(mean_diff, ls="--", color="#84142D")

    vert_text(ax, f"Lower bound = {lb:.3f}", lb)
    vert_text(ax, f"Upper bound = {ub:.3f}", ub)
    vert_text(ax, f"Sample mean difference = {mean_diff:.3f}", mean_diff, "#84142D")

    kde = ax.get_lines()[0].get_data()
    ax.fill_between(
        kde[0],
        kde[1],
        where=(kde[0] > lb) & (kde[0] < ub),
        interpolate=True,
        color="#B6C9F0",
    )

    ax.set_title(
        "Bootstraped distribution of difference in means\nwith 95% CI", fontsize=15
    )
    plt.show()


def plot_pvalue(
    dist: np.ndarray, mean_diff: float, p_value: float, alt="two_sided"
) -> None:
    """
    Visualize the distribution of permuted mean differences
    Takes in values from permute_p
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.kdeplot(dist)
    ax.axvline(mean_diff, ls="--", color="b")
    vert_text(ax, "Sample mean difference", mean_diff)
    kde = ax.get_lines()[0].get_data()
    ax.set_title("Permuted distribution of difference in means", fontsize=15)

    if alt == "two_sided":
        ax.axvline(-mean_diff, ls="--", color="b")
        vert_text(ax, "-Sample mean difference", -mean_diff)
        ax.fill_between(
            kde[0],
            kde[1],
            where=(kde[0] < -abs(mean_diff)) | (kde[0] > abs(mean_diff)),
            interpolate=True,
            color="#FF7272",
        )
    elif alt == "greater":
        ax.fill_between(
            kde[0],
            kde[1],
            where=(kde[0] > mean_diff),
            interpolate=True,
            color="#FF7272",
        )
    elif alt == "less":
        ax.fill_between(
            kde[0],
            kde[1],
            where=(kde[0] < mean_diff),
            interpolate=True,
            color="#FF7272",
        )
    else:
        raise ValueError("alternative must be " "'less', 'greater' or 'two_sided'")
    plt.show()
    
    
def compare_cf_matrix(
    model: Callable,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    size: tuple=(8,4),
    normalize: str='true'
) -> None:
    """
    Plots confusion matrices for both train and test sets
    to see how well the model is performing and if it's not
    overfitting
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(1, 2, figsize=size, sharey=True)

    ConfusionMatrixDisplay.from_estimator(
        model, X_train, y_train, normalize=normalize, colorbar=False, ax=ax[0]
    )
    ax[0].set_title("Training confusion matrix")
    ax[0].grid(False)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, normalize=normalize, colorbar=False, ax=ax[1]
    )
    ax[1].set_title("Testing confusion matrix")
    ax[1].grid(False)


#####################
# Statistical tests #
#####################


def calc_conf_pool(data1: pd.Series, data2: pd.Series, t: float) -> tuple[float]:
    """
    Implementation of a pooled confidence interval calculation for difference
    in two means.
    """
    mean_diff = data1.mean() - data2.mean()
    n1 = data1.size
    n2 = data2.size

    num = (n1 - 1) * data1.std() ** 2 + (n2 - 1) * data2.std() ** 2
    den = n1 + n2 - 2

    lb = mean_diff - t * (np.sqrt(num / den) * np.sqrt(1 / n1 + 1 / n2))
    ub = mean_diff + t * (np.sqrt(num / den) * np.sqrt(1 / n1 + 1 / n2))
    return lb, mean_diff, ub


def boot_mean_diff_ci(
    data1: pd.Series, data2: pd.Series, n_boot: int, alpha: float = 0.95
) -> tuple[float, float, np.ndarray]:
    """
    Bootstrap implementation to calculate confidence intervals between two sample means
    """
    mean_diff = data1.mean() - data2.mean()
    diffs = np.zeros((n_boot,))
    for i in range(n_boot):
        mean1 = np.random.choice(data1, data1.size).mean()
        mean2 = np.random.choice(data2, data2.size).mean()
        diffs[i] = mean1 - mean2
    quant = (1 - alpha) / 2
    lb = np.quantile(diffs, quant)
    ub = np.quantile(diffs, 1 - quant)
    return lb, mean_diff, ub, diffs


def permute_p(
    data1: np.ndarray, data2: np.ndarray, n_perm: int, alt: str = "two_sided"
) -> tuple[float, float, np.ndarray]:
    """
    Calculates p-value between two datasets means using permutations
    """
    mean_diff = data1.mean() - data2.mean()
    comb = np.concatenate((data1, data2))
    diffs = np.zeros((n_perm,))

    for i in range(n_perm):
        shuffled = np.random.permutation(comb)
        mean1 = shuffled[: data1.size].mean()
        mean2 = shuffled[data1.size :].mean()
        diffs[i] = mean1 - mean2

    if alt == "two_sided":
        p_value = np.mean(np.abs(diffs) >= np.abs(mean_diff))
    elif alt == "greater":
        p_value = np.mean(diffs >= mean_diff)
    elif alt == "less":
        p_value = np.mean(diffs <= mean_diff)
    else:
        raise ValueError("alternative must be " "'less', 'greater' or 'two_sided'")

    return p_value, mean_diff, diffs


######################
# Analysis functions #
######################


def rmse(pred_y: np.ndarray, true_y: np.ndarray) -> float:
    """
    Calculates the root mean square error between two arrays of values
    """
    assert len(pred_y) == len(true_y), "Different length target and predictions"
    diff = pred_y - true_y
    length = len(pred_y)
    return np.sqrt((diff ** 2).sum() / length)


######################
# Tunning stuff      #
######################

param_dict = dict
sklearn_model_name = str
sklearn_model = Callable


class EstimatorSelectionHelper:
    """
    Classs to automate screening for multiple models with
    grid searches.
    """

    def __init__(
        self,
        #models: dict[sklearn_model_name, sklearn_model],
        params: dict[sklearn_model_name, param_dict],
        pipe: Pipeline,
    ) -> None:
        #if not set(models.keys()).issubset(set(params.keys())):
        #    missing_params = list(set(models.keys()) - set(params.keys()))
        #    raise ValueError(
        #        "Some estimators are missing parameters: %s" % missing_params
        #    )
        #self.models = models
        self.params = params
        self.pipe = pipe
        self.grid_searches = {}

    def make_random_searches(
        self,
        cv: int = 3,
        n_jobs: int = -1,
        verbose: int = 1,
        n_iter: int = 25,
        scoring: str = None,
        refit: bool = False,
    ) -> None:
        """
        Creates RandomSearchCV instances before fitting them.
        Also has a side effect of checking if all params are
        specified correctly.
        """
        for key in self.params.keys():
            #model = self.models[key]
            params = self.params[key]
            grid = RandomizedSearchCV(
                self.pipe,
                params,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=refit,
                return_train_score=True,
            )
            self.grid_searches[key] = grid

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fits all the GridSearchCV instances.
        """
        self.make_random_searches(**kwargs)
        for key in self.params.keys():
            print("Running GridSearchCV for %s." % key)
            self.grid_searches[key].fit(X, y)

    def score_summary(self, sort_by: str = "mean_score") -> pd.DataFrame:
        """
        Creates and returns a DataFrame with model scores
        and their given parameters.
        """
        
        def make_row(key: str, scores: np.ndarray, params: dict) -> pd.Series:
            """
            Creates Series for each trained model instance
            with its scores.
            """
            d = {
                "estimator": key,
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
            }
            d['params'] = params
            return pd.Series({**d})
        
        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_["params"]
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((make_row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        columns = ["estimator", "min_score", "mean_score", "max_score", "std_score"]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
    
from sklearn.base import BaseEstimator, ClassifierMixin


class MyEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

    def predict_proba(self, X):
        probas = []
        for i, model in enumerate(self.models):
            pred = model.predict_proba(X)[:, 1]
            probas.append(pred)
        return np.array(probas).mean(axis=0)


    
