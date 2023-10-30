from __future__ import annotations

from typing import Union

import meerkat as mk
import numpy as np

from sklearn.decomposition import PCA

from domino.utils import convert_to_numpy, unpack_args
from ._generalized import GeneralizedMixture, GaussianVariable, CategoricalVariable
from ..abstract import Slicer


class MixtureSlicer(Slicer):

    r"""
    Slice Discovery based on the Domino Mixture Model.

    Discover slices by jointly modeling a mixture of input embeddings (e.g. activations
    from a trained model), class labels, and model predictions. This encourages slices
    that are homogeneous with respect to error type (e.g. all false positives).

    Examples
    --------
    Suppose you've trained a model and stored its predictions on a dataset in
    a `Meerkat DataFrame <https://github.com/robustness-gym/meerkat>`_ with columns
    "emb", "target", and "pred_probs". After loading the DataFrame, you can discover
    underperforming slices of the validation dataset with the following:

    .. code-block:: python

        from domino import MixtureSlicer
        dp = ...  # Load dataset into a Meerkat DataFrame

        # split dataset
        valid_dp = dp[dp["split"] == "valid"]
        test_dp = dp[dp["split"] == "test"]

        domino = MixtureSlicer()
        domino.fit(
            data=valid_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        dp["domino_slices"] = domino.predict(
            data=test_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )


    Args:
        n_slices (int, optional): The number of slices to discover.
            Defaults to 5.
        covariance_type (str, optional): The type of covariance parameter
            :math:`\mathbf{\Sigma}` to use. Same as in sklearn.mixture.GaussianMixture.
            Defaults to "diag", which is recommended.
        n_pca_components (Union[int, None], optional): The number of PCA components
            to use. If ``None``, then no PCA is performed. Defaults to 128.
        n_mixture_components (int, optional): The number of clusters in the mixture
            model, :math:`\bar{k}`. This differs from ``n_slices`` in that the
            ``MixtureSlicer`` only returns the top ``n_slices`` with the highest error rate
            of the ``n_mixture_components``. Defaults to 25.
        y_log_likelihood_weight (float, optional): The weight :math:`\gamma` applied to
            the :math:`P(Y=y_{i} | S=s)` term in the log likelihood during the E-step.
            Defaults to 1.
        y_hat_log_likelihood_weight (float, optional): The weight :math:`\hat{\gamma}`
            applied to the :math:`P(\hat{Y} = h_\theta(x_i) | S=s)` term in the log
            likelihood during the E-step. Defaults to 1.
        max_iter (int, optional): The maximum number of iterations to run. Defaults
            to 100.
        init_params (str, optional): The initialization method to use. Options are
            the same as in sklearn.mixture.GaussianMixture plus one addition,
            "confusion". If "confusion",  the clusters are initialized such that almost
            all of the examples in a cluster come from same cell in the confusion
            matrix. See Notes below for more details. Defaults to "confusion".
        confusion_noise (float, optional): Only used if ``init_params="confusion"``.
            The scale of noise added to the confusion matrix initialization. See notes
            below for more details.
            Defaults to 0.001.
        random_state (Union[int, None], optional): The random seed to use when
            initializing  the parameters.

    Notes
    -----

    The mixture model is an extension of a standard Gaussian Mixture Model. The model is
    based on the assumption that data is generated according to the following generative
    process.

    * Each example belongs to one of :math:`\bar{k}` slices. This slice
      :math:`S` is sampled from a categorical
      distribution :math:`S \sim Cat(\mathbf{p}_S)` with parameter :math:`\mathbf{p}_S
      \in\{\mathbf{p} \in \mathbb{R}_+^{\bar{k}} : \sum_{i = 1}^{\bar{k}} p_i = 1\}`
      (see ``MixtureSlicer.mm.weights_``).
    * Given the slice :math:`S'`, the embeddings are normally distributed
      :math:`Z | S \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma}`)  with parameters
      mean :math:`\mathbf{\mu} \in \mathbb{R}^d` (see ``MixtureSlicer.mm.means_``) and
      :math:`\mathbf{\Sigma} \in \mathbb{S}^{d}_{++}`
      (see ``MixtureSlicer.mm.covariances_``;
      normally this parameter is constrained to the set of symmetric positive definite
      :math:`d \\times d` matrices, however the argument ``covariance_type`` allows for
      other constraints).
    * Given the slice, the labels vary as a categorical
      :math:`Y |S \sim Cat(\mathbf{p})` with parameter :math:`\mathbf{p}
      \in \{\mathbf{p} \in \mathbb{R}^c_+ : \sum_{i = 1}^c p_i = 1\}` (see
      ``MixtureSlicer.mm.y_probs``).
    * Given the slice, the model predictions also vary as a categorical
      :math:`\hat{Y} | S \sim Cat(\mathbf{\hat{p}})` with parameter
      :math:`\mathbf{\hat{p}} \in \{\mathbf{\hat{p}} \in \mathbb{R}^c_+ :
      \sum_{i = 1}^c \hat{p}_i = 1\}` (see ``MixtureSlicer.mm.y_hat_probs``).

    The mixture model is, thus, parameterized by :math:`\phi = [\mathbf{p}_S, \mu,
    \Sigma, \mathbf{p}, \mathbf{\hat{p}}]` corresponding to the attributes
    ``weights_, means_, covariances_, y_probs, y_hat_probs`` respectively. The
    log-likelihood over the :math:`n` examples in the validation dataset :math:`D_v` is
    given as followsand maximized using expectation-maximization:

    .. math::
        \ell(\phi) = \sum_{i=1}^n \log \sum_{s=1}^{\hat{k}} P(S=s)P(Z=z_i| S=s)
        P( Y=y_i| S=s)P(\hat{Y} = h_\theta(x_i) | S=s)

    We include two optional hyperparameters
    :math:`\gamma, \hat{\gamma} \in \mathbb{R}_+`
    (see ``y_log_liklihood_weight`` and ``y_hat_log_likelihood_weight`` below) that
    balance the importance of modeling the class labels and predictions against the
    importance of modeling the embedding. The modified log-likelihood over :math:`n`
    examples is given as follows:

    .. math::
        \ell(\phi) = \sum_{i=1}^n \log \sum_{s=1}^{\hat{k}} P(S=s)P(Z=z_i| S=s)
        P( Y=y_i| S=s)^\gamma P(\hat{Y} = h_\theta(x_i) | S=s)^{\hat{\gamma}}

    .. attention::
        Although we model the prediction :math:`\hat{Y}` as a categorical random
        variable, in practice predictions are sometimes "soft" (e.g. the output
        of a softmax layer is a probability distribution over labels, not a single
        label). In these cases, the prediction :math:`\hat{Y}` is technically a
        dirichlet random variable (i.e. a distribution over distributions).

        However, to keep the implementation simple while still leveraging the extra
        information provided by "soft" predictions, we naïvely plug the "soft"
        predictions directly into the categorical PMF in the E-step and the update in
        the M-step. Specifically, during the E-step, instead of computing the
        categorical PMF :math:`P(\hat{Y}=\hat{y_i} | S=s)` we compute
        :math:`\sum_{j=1}^c \hat{y_i}(j) P(\hat{Y}=j | S=s)` where :math:`\hat{y_i}(j)`
        is the "soft" prediction for class :math:`j` (we can
        think of this like we're marginalizing out the uncertainty in the prediction).
        During the M-step, we compute a "soft" update for the categorical parameters
        :math:`p_j^{(s)} = \sum_{i=1}^n Q(s,i) \hat{y_i}(j)` where :math:`Q(s,i)`
        is the "responsibility" of slice :math:`s` towards the data point :math:`i`.

    When using ``"confusion"`` initialization, each slice $s^{(j)}$ is assigned a
    :math:`y^{(j)}\in \mathcal{Y}` and :math:`\hat{y}^{(j)} \in \mathcal{Y}` (*i.e.*
    each slice is assigned a cell in the confusion matrix). This is typically done in a
    round-robin fashion so that there are at least
    :math:`\floor{\hat{k} / {|\mathcal{Y}|^2}}`
    slices assigned to each cell in the confusion matrix. Then, we fill in the initial
    responsibility matrix :math:`Q \in \mathbb{R}^{n \times \hat{k}}`, where each cell
    :math:`Q_{ij}` corresponds to our model's initial estimate of
    :math:`P(S=s^{(j)}|Y=y_i,
    \hat{Y}=\hat{y}_i)`. We do this according to

    .. math::
        \bar{Q}_{ij} \leftarrow
        \begin{cases}
            1 + \epsilon & y_i=y^{(j)} \land \hat{y}_i = \hat{y}^{(j)} \\
            \epsilon & \text{otherwise}
        \end{cases}

    .. math::
        Q_{ij} \leftarrow \frac{\bar{Q}_{ij} } {\sum_{l=1}^{\hat{k}} \bar{Q}_{il}}

    where :math:`\epsilon` is random noise which ensures that slices assigned to the
    same confusion matrix cell won't have the exact same initialization. We sample
    :math:`\epsilon` uniformly from the range ``(0, confusion_noise]``.

    """

    def __init__(
        self,
        n_slices: int = 5,
        covariance_type: str = "diag",
        n_pca_components: Union[int, None] = 128,
        n_mixture_components: int = 25,
        y_log_likelihood_weight: float = 1,
        y_hat_log_likelihood_weight: float = 1,
        max_iter: int = 100,
        init_params: str = "kmeans",
        confusion_noise: float = 1e-3,
        random_state: int = None,
        pbar: bool = True,
    ):
        super().__init__(n_slices=n_slices)

        self.config.covariance_type = covariance_type
        self.config.n_pca_components = n_pca_components
        self.config.n_mixture_components = n_mixture_components
        self.config.init_params = init_params
        self.config.confusion_noise = confusion_noise
        self.config.y_log_likelihood_weight = y_log_likelihood_weight
        self.config.y_hat_log_likelihood_weight = y_hat_log_likelihood_weight
        self.config.max_iter = max_iter
        self.config.random_state = random_state

        if self.config.n_pca_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=self.config.n_pca_components)

    def fit(
        self,
        data: Union[dict, mk.DataFrame] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = None,
    ) -> MixtureSlicer:
        """
        Fit the mixture model to data.

        Args:
            data (mk.DataFrame, optional): A `Meerkat DataFrame` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".

        Returns:
            MixtureSlicer: Returns a fit instance of MixtureSlicer.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        variables = [
            GaussianVariable(
                name="embeddings",
                weight=1,
                covariance_type=self.config.covariance_type,
            )
        ]
        if targets is not None:
            variables.append(
                CategoricalVariable(
                    name="targets",
                    weight=self.config.y_log_likelihood_weight,
                )
            )
        if pred_probs is not None:
            variables.append(
                CategoricalVariable(
                    name="pred_probs",
                    weight=self.config.y_hat_log_likelihood_weight,
                )
            )
        if losses is not None:
            variables.append(
                GaussianVariable(
                    name="losses",
                    weight=1,
                    covariance_type=self.config.covariance_type,
                )
            )

        self.mm = GeneralizedMixture(
            n_components=self.config.n_mixture_components,
            init_params=self.config.init_params,
            variables=variables,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )

        if self.pca is not None:
            self.pca.fit(X=embeddings)
            embeddings = self.pca.transform(X=embeddings)

        self.mm.fit(embeddings=embeddings, targets=targets, pred_probs=pred_probs)

        # TODO: fix this: it's a hack!
        if len(self.mm.variables) == 2:
            self.slice_cluster_indices =  -np.abs(
                (self.mm.variables[1].probs).max(axis=1)
            ).argsort()[: self.config.n_slices]
        else:
            self.slice_cluster_indices = (
                -np.abs(
                    (self.mm.variables[2].probs - self.mm.variables[1].probs).max(axis=1)
                )
            ).argsort()[: self.config.n_slices]
        return self

    def predict(
        self,
        data: Union[dict, mk.DataFrame] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = "losses",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.


        .. caution::
            Must call ``MixtureSlicer.fit`` prior to calling ``MixtureSlicer.predict``.


        Args:
            data (mk.DataFrame, optional): A `Meerkat DataFrame` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): Ignored.

        Returns:
            np.ndarray: A binary ``np.ndarray`` of shape (n_samples, n_slices) where
                values are either 1 or 0.
        """
        probs = self.predict_proba(
            data=data,
            embeddings=embeddings,
            targets=targets,
            pred_probs=pred_probs,
        )
        preds = np.zeros_like(probs, dtype=np.int32)
        preds[np.arange(preds.shape[0]), probs.argmax(axis=-1)] = 1
        return preds

    def predict_proba(
        self,
        data: Union[dict, mk.DataFrame] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = "loss",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.

        .. caution::
            Must call ``MixtureSlicer.fit`` prior to calling
            ``MixtureSlicer.predict_proba``.


        Args:
            data (mk.DataFrame, optional): A `Meerkat DataFrame` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): Ignored.

        Returns:
            np.ndarray: A ``np.ndarray`` of shape (n_samples, n_slices) where values in
                are in range [0,1] and rows sum to 1.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        if self.pca is not None:
            embeddings = self.pca.transform(X=embeddings)

        clusters = self.mm.predict_proba(
            embeddings=embeddings, targets=targets, pred_probs=pred_probs
        )

        return clusters[:, self.slice_cluster_indices]


DominoSlicer = MixtureSlicer
