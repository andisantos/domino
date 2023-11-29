"""Generalized Mixture Model.

This is based closely off the scikit-learn implementation of the Gaussian
mixture model by author: Wei Xue <xuewei4d@gmail.com>.
"""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

from tkinter import Variable
from typing import List
import warnings
import numpy as np

from scipy import linalg

from ._base import BaseMixture, _check_shape, _check_X, check_random_state

from scipy import linalg
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm


###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class


def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")

    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )

    # check normalization
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError(
            "The parameter 'weights' should be normalized, "
            "but got sum(weights) = %.5f" % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be " "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (
        np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
    ):
        raise ValueError(
            "'%s precision' should be symmetric, " "positive-definite" % covariance_type
        )


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(
        precisions,
        dtype=[np.float64, np.float32],
        ensure_2d=False,
        allow_nd=covariance_type == "full",
    )

    precisions_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }
    _check_shape(
        precisions, precisions_shape[covariance_type], "%s precision" % covariance_type
    )

    _check_precisions = {
        "full": _check_precisions_full,
        "tied": _check_precision_matrix,
        "diag": _check_precision_positivity,
        "spherical": _check_precision_positivity,
    }
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )

    elif covariance_type == "tied":
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        precisions = precisions_chol**2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )

    elif covariance_type == "spherical":
        precisions = precisions_chol**2
        log_prob = (
            np.sum(means**2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class Variable:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def preprocess(self, value: np.ndarray):
        return value

    def check(self, value: np.ndarray, n_components: int):
        return value


class CategoricalVariable(Variable):
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
    ):
        super().__init__(name, weight)

    def preprocess(self, value: np.ndarray):
        y = value
        # we want to support continuous binary labels as well
        if y.dtype == np.dtype(int):
            y = label_binarize(y, classes=np.arange(np.max(y) + 1))
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if y.shape[-1] == 1:
            # binary targets transform to a column vector with label_binarize
            y = np.array([1 - y[:, 0], y[:, 0]]).T
        return y

    def estimate_parameters(self, value: np.ndarray, resp: np.ndarray, nk: np.ndarray):
        return np.dot(resp.T, value) / nk[:, np.newaxis]

    def initialize_parameters(
        self, value: np.ndarray, resp: np.ndarray, nk: np.ndarray = None
    ):
        if nk is None:
            nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.probs = self.estimate_parameters(value, resp, nk)

    def update_parameters(
        self, value: np.ndarray, resp: np.ndarray, nk: np.ndarray = None
    ):
        if nk is None:
            nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.probs = self.estimate_parameters(value, resp, nk)

    def estimate_log_prob(self, value: np.ndarray):
        # add epsilon to avoid "RuntimeWarning: divide by zero encountered in log"
        if (np.dot(value, self.probs.T) + np.finfo(self.probs.dtype).eps < 0).any():
            import pdb

            pdb.set_trace()

        return np.log(np.dot(value, self.probs.T) + np.finfo(self.probs.dtype).eps)

    def get_parameters(self):
        return (self.probs,)

    def set_parameters(self, params):
        self.probs = params[0]
        return params


class GaussianVariable(Variable):
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        covariance_type: int = "full",
        reg_covar: float = 1e-6,
        means_init: np.ndarray = None,
        precisions_init: np.ndarray = None,
    ):
        super().__init__(name, weight)
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.means_init = means_init
        self.precisions_init = precisions_init

    def initialize_parameters(
        self, value: np.ndarray, resp: np.ndarray, nk: np.ndarray = None
    ):
        if nk is None:
            nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # (n_components, )

        means, covariances = self.estimate_parameters(value, resp, nk)

        self.means_ = means if self.means_init is None else self.means_init
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = self.precisions_init

    def update_parameters(
        self, value: np.ndarray, resp: np.ndarray, nk: np.ndarray = None
    ):
        if nk is None:
            nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

        self.means_, self.covariances_ = self.estimate_parameters(value, resp, nk)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def estimate_parameters(self, value: np.ndarray, resp: np.ndarray, nk: np.ndarray):
        means = np.dot(resp.T, value) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": _estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[self.covariance_type](resp, value, nk, means, self.reg_covar)
        return means, covariances

    def estimate_log_prob(self, value: np.ndarray):
        return _estimate_log_gaussian_prob(
            value, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    def check(self, value: np.ndarray, n_components: int):
        value = _check_X(value, n_components=n_components, ensure_min_samples=2)
        self._check_n_features(value, reset=True)

        if self.reg_covar < 0.0:
            raise ValueError(
                "Invalid value for 'reg_covar': %.5f "
                "regularization on covariance must be "
                "non-negative" % self.reg_covar
            )

        self._check_parameters(value, n_components=n_components)
        return value

    def _check_parameters(self, X: np.ndarray, n_components: int):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']" % self.covariance_type
            )

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def get_parameters(self):
        return (
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def set_parameters(self, params):
        self.means_, self.covariances_, self.precisions_cholesky_ = params
        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_**2

    def _check_n_features(self, X: np.ndarray, reset: bool = False):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
                It is recommended to call reset=True in `fit` and in the first
                call to `partial_fit`. All other methods that validate `X`
                should set `reset=False`.
        """
        try:
            n_features = X.shape[1]
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                raise ValueError(
                    "X does not contain any features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features"
                ) from e
            # If the number of features is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )


class GeneralizedMixture(BaseMixture):
    def __init__(
        self,
        n_components: int = 1,
        variables: List[Variable] = None,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: int = "kmeans",
        tol: float = 1e-3,
        random_state: np.ndarray = None,
        weights_init: np.ndarray = None,
        warm_start=False,
        verbose: int = 0,
        verbose_interval: int = 10,
        confusion_noise: float = 1e-3,
        pbar: bool = True,
    ):
        if variables is None:
            variables = [Variable("X", "gaussian")]
        self.variables = variables
        self.confusion_noise = confusion_noise
        self.pbar = pbar
        self.weights_init = weights_init

        super().__init__(
            n_components=n_components,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
            reg_covar=None,  # for compatibility with superclass, but moved to variable
        )

    def _initialize_parameters(self, values: List[np.ndarray], random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        X = values[0]  # TODO(sabri): make this work for any number of variables
        n_samples, _ = values[0].shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )

        self._initialize(values=values, resp=resp)

    def _initialize(self, values: List[np.ndarray], resp: np.ndarray):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = values[0].shape

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # (n_components, )
        for value, variable in zip(values, self.variables):
            variable.initialize_parameters(value, resp, nk)

        weights = nk / n_samples
        self.weights_ = weights if self.weights_init is None else self.weights_init

    def fit(self, **kwargs):
        self.fit_predict(**kwargs)
        return self

    def predict(self, **kwargs):
        return self.predict_proba(**kwargs).argmax(axis=1)

    def predict_proba(self, **kwargs):
        values = [kwargs.get(variable.name, None) for variable in self.variables]
        values = self._preprocess(values)
        values = self._check(values)

        check_is_fitted(self)
        _, log_resp = self._estimate_log_prob_resp(values)
        return np.exp(log_resp)

    def _preprocess(self, values: List[np.ndarray]):
        out = []
        for value, variable in zip(values, self.variables):
            if value is None:
                out.append(None)
            else:
                out.append(variable.preprocess(value))
        return out

    def _check(self, values: List[np.ndarray]):
        if self.tol < 0.0:
            raise ValueError(
                "Invalid value for 'tol': %.5f "
                "Tolerance used by the EM must be non-negative" % self.tol
            )

        if self.n_init < 1:
            raise ValueError(
                "Invalid value for 'n_init': %d "
                "Estimation requires at least one run" % self.n_init
            )

        if self.max_iter < 1:
            raise ValueError(
                "Invalid value for 'max_iter': %d "
                "Estimation requires at least one iteration" % self.max_iter
            )

        out = []
        for value, variable in zip(values, self.variables):
            if value is None:
                out.append(None)
            out.append(variable.check(value, self.n_components))
        return out

    def fit_predict(self, **kwargs):
        if not all(variable.name in kwargs for variable in self.variables):
            raise ValueError("All variables must be present in the data passed to fit.")

        values = [kwargs.get(variable.name, None) for variable in self.variables]

        values = self._preprocess(values)

        values = self._check(values)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        best_params = None
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(values, random_state)

            lower_bound = -np.infty if do_init else self.lower_bound_

            for n_iter in tqdm(
                range(1, self.max_iter + 1), disable=not self.pbar
            ):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(values)
                if np.isnan(log_resp).any():
                    import pdb

                    pdb.set_trace()

                self._m_step(values, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        if best_params is None:
            self._initialize_parameters(values, random_state)
        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(values)

        return log_resp.argmax(axis=1)

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    def _m_step(self, values, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = values[0].shape
        resp = np.exp(log_resp)

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # (n_components, )
        for value, variable in zip(values, self.variables):
            variable.update_parameters(value, resp, nk)

        self.weights_ = nk / n_samples

    def _e_step(self, values: List[np.ndarray]):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(values)
        return np.mean(log_prob_norm), log_resp

    def _estimate_log_prob_resp(self, values: List[np.ndarray]):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(values)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, values: List[np.ndarray]):
        log_prob = self._estimate_log_weights()
        for value, variable in zip(values, self.variables):
            if value is None:
                continue  # skip values not provided
            log_prob = variable.estimate_log_prob(value) * variable.weight + log_prob

        return log_prob

    def _set_parameters(self, params):
        self.weights_ = params.pop(-1)
        for variable, param in zip(self.variables, params):
            variable.set_parameters(param)

    def _get_parameters(self):
        params = [variable.get_parameters() for variable in self.variables]
        params.append(self.weights_)
        return params

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_parameters(self, X):
        raise NotImplementedError
