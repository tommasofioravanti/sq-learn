""" Quantum Principal Component Analysis.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <denis-alexander.engemann@inria.fr>
#         Michael Eickenberg <michael.eickenberg@inria.fr>
#         Giorgio Patrini <giorgio.patrini@anu.edu.au>
#
# License: BSD 3 clause
from math import log, sqrt
import numbers
from scipy import linalg
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
import matlab.engine
import matlab
from ._base import _BasePCA
from ..utils import check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils.extmath import fast_logdet, randomized_svd, svd_flip
from ..utils.extmath import stable_cumsum
from ..utils.validation import check_is_fitted
from ..utils.validation import _deprecate_positional_args
from ..QuantumUtility.Utility import *


def _assess_dimension(spectrum, rank, n_samples):
    """Compute the log-likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``. This implements the method of
    T. P. Minka.

    Parameters
    ----------
    spectrum : ndarray of shape (n_features,)
        Data spectrum.
    rank : int
        Tested rank value. It should be strictly lower than n_features,
        otherwise the method isn't specified (division by zero in equation
        (31) from the paper).
    n_samples : int
        Number of samples.

    Returns
    -------
    ll : float
        The log-likelihood.

    References
    ----------
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    <https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf>`_
    """

    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:
        raise ValueError("the tested rank should be in [1, n_features - 1]")

    eps = 1e-15

    if spectrum[rank - 1] < eps:
        # When the tested rank is associated with a small eigenvalue, there's
        # no point in computing the log-likelihood: it's going to be very
        # small and won't be the max anyway. Also, it can lead to numerical
        # issues below when computing pa, in particular in log((spectrum[i] -
        # spectrum[j]) because this will take the log of something very small.
        return -np.inf

    pu = -rank * log(2.)
    for i in range(1, rank + 1):
        pu += (gammaln((n_features - i + 1) / 2.) -
               log(np.pi) * (n_features - i + 1) / 2.)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    v = max(eps, np.sum(spectrum[rank:]) / (n_features - rank))
    pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll


def _infer_dimension(spectrum, n_samples):
    """Infers the dimension of a dataset with a given spectrum.

    The returned value will be in [1, n_features - 1].
    """
    ll = np.empty_like(spectrum)
    ll[0] = -np.inf  # we don't want to return n_components = 0
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return ll.argmax()


class qPCA(_BasePCA):
    """Quantum Principal component analysis (qPCA).

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    It uses the LAPACK implementation of the full SVD or a randomized truncated
    SVD by the method of Halko et al. 2009, depending on the shape of the input
    data and the number of components to extract.

    It can also use the scipy.sparse.linalg ARPACK implementation of the
    truncated SVD.

    Notice that this class does not support sparse input. See
    :class:`TruncatedSVD` for an alternative with sparse data.

    Notice also that the quantum routines are implemented only for the 'full' svd_solver case.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

        .. versionadded:: 0.18.0

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

        .. versionadded:: 0.18.0

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

        .. versionadded:: 0.18.0

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 0.18.0

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        The variance estimation uses `n_samples - 1` degrees of freedom.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

        .. versionadded:: 0.18

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

        .. versionadded:: 0.19

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    See Also
    --------
    KernelPCA : Kernel Principal Component Analysis.
    SparsePCA : Sparse Principal Component Analysis.
    TruncatedSVD : Dimensionality reduction using truncated SVD.
    IncrementalPCA : Incremental Principal Component Analysis.

    References
    ----------
    For n_components == 'mle', this class uses the method from:
    `Minka, T. P.. "Automatic choice of dimensionality for PCA".
    In NIPS, pp. 598-604 <https://tminka.github.io/papers/pca/minka-pca.pdf>`_

    Implements the probabilistic PCA model from:
    `Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
    component analysis". Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 61(3), 611-622.
    <http://www.miketipping.com/papers/met-mppca.pdf>`_
    via the score and score_samples methods.

    For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

    For svd_solver == 'randomized', see:
    `Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions".
    SIAM review, 53(2), 217-288.
    <https://doi.org/10.1137/090771806>`_
    and also
    `Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
    "A randomized algorithm for the decomposition of matrices".
    Applied and Computational Harmonic Analysis, 30(1), 47-68
    <https://doi.org/10.1016/j.acha.2010.02.003>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(n_components=2)
    >>> print(pca.explained_variance_ratio_)
    [0.9924... 0.0075...]
    >>> print(pca.singular_values_)
    [6.30061... 0.54980...]

    >>> pca = PCA(n_components=2, svd_solver='full')
    >>> pca.fit(X)
    PCA(n_components=2, svd_solver='full')
    >>> print(pca.explained_variance_ratio_)
    [0.9924... 0.00755...]
    >>> print(pca.singular_values_)
    [6.30061... 0.54980...]

    >>> qpca = qPCA(svd_solver='full')
    qPCA fit with eps and delta error to estimate factor_score, singular values ecc... .
    >>> qpca.fit(X,eps=0.1,theta_estimate=True,eps_theta=0.05,p=0.70,estimate_all=True,delta = 0.5)
    >>> print(qpca.estimate_fs_ratio)
    [1.0]
    >>> print(qpca.estimate_s_values)
    [1.07791089]

    >>> pca = PCA(n_components=1, svd_solver='arpack')
    >>> pca.fit(X)
    PCA(n_components=1, svd_solver='arpack')
    >>> print(pca.explained_variance_ratio_)
    [0.99244...]
    >>> print(pca.singular_values_)
    [6.30061...]
    """

    @_deprecate_positional_args
    def __init__(self, n_components=None, *, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None, name=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.name = name
        if random_state:
            random.seed(random_state)
        self.quantum_runtime_container = []

    def fit(self, X, y=None, quantum_retained_variance=False, eps=0,
            theta_major=0,
            theta_minor=0, eta=0, theta_estimate=False, use_computed_qcomponents=False, eps_theta=0, p=0,
            estimate_all=False, delta=0, true_tomography=True, fs_ratio_estimation=False, norm='L2',
            stop_when_reached_accuracy=False, incremental_measure=False, faster_measure_increment=0,
            check_sv_uniform_distribution=False, spectral_norm_est=False, condition_number_est=False,
            estimate_least_k=False):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        quantum_retained_variance : bool, default=False.
            If true it computes the retained variance in the quantum version of the algorithm (Theorem 9 of QADRA)

        eps: float, default=0.
            Error to introduce for the singular values estimation.

        theta_major: float, default=0.
            Smallest singular values to retain in order to compute the retained variance.

        theta_minor: float, default=0.
            Greatest singular values to retain. It is very useful in case one want to extract and estimate the least
            singular values: with the right theta_minor value, we are able to select all the estimated singular values
            that are less than this value.

        eta: float, default=0.
            Used to compute the relative error of the estimated variance. If it is zero,
            the retained variance is returned without any error.

        theta_estimate: bool, default=False.
            If true compute the estimation of theta using quantum binary search.

        eps_theta: float, default=0.
            Error to introduce in the estimation of theta in SVE in quantum binary search.

        p: float, default=0.
            Percentage of retained variance to estimate theta.

        estimate_all: bool, default=False.
            If true, singular vectors (left and right), singular values and factor scores are estimated.

        delta: float, default=0.
            Is the error to insert in the estimation of the singular vectors.

        true_tomography: bool, default=True.
            If true means that the quantum estimations are are done with real tomography,
            otherwise the estimations are approximated with a Truncated Gaussian Noise.

        fs_ratio_estimation: bool, default=False.
            If true, it estimates factor score ratios using SVE.

        norm: string, default='L2'.
            If 'L2' (and true_tomography is True):
                true_tomography is executed with L2-norm guarantees.
            If 'inf' (and true_tomography is True):
                true_tomography is executed with Linf-norm guarantees.

        stop_when_reached_accuracy: bool, default=True.
            If True, the execution of the tomography is stopped when we reach an estimate delta-close in L2-norm
            to the true value that we are estimating. Otherwise all the N=36d log(d)/delta**2 measures are performed.

        incremental_measure: bool, default=True.
            If True, tomography is performed many times with different number of measures. If False, the routine is
            performed once using exactly N=36d log(d)/delta**2 measures.

        faster_measure_increment: int, default=0.
            It increments the tomography measures of a specific constant value. It is useful in the case one want to
            speed-up the execution of the tomography.

        spectral_norm_est: bool, default=False.
            If True, an estimation of the spectral norm of the input matrix is computed.

        estimate_least_k: bool, default=False.
            If True, the quantum least singular vectors extractor is executed. In this case it is important to pass a sensible
            theta_minor parameter to consider the right singular values.


        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if quantum_retained_variance:
            if eps <= 0:
                raise ValueError("eps must be > 0")
            if theta_major <= 0 and theta_estimate == False:
                raise ValueError("theta must be > 0")
        if theta_estimate:
            if p <= 0 and not isinstance(self.n_components, int):
                raise ValueError("p must be > 0")

        self._fit(X,
                  quantum_retained_variance=quantum_retained_variance, eps=eps, theta_major=theta_major,
                  theta_minor=theta_minor,
                  eta=eta, theta_estimate=theta_estimate, use_computed_qcomponents=use_computed_qcomponents,
                  eps_theta=eps_theta, ret_var=p, estimate_all=estimate_all, delta=delta,
                  true_tomography=true_tomography,
                  fs_ratio_estimation=fs_ratio_estimation, norm=norm,
                  stop_when_reached_accuracy=stop_when_reached_accuracy, incremental_measure=incremental_measure,
                  faster_measure_increment=faster_measure_increment,
                  check_sv_uniform_distribution=check_sv_uniform_distribution, spectral_norm_est=spectral_norm_est,
                  condition_number_est=condition_number_est, estimate_least_k=estimate_least_k)
        return self

    def fit_transform(self, X, y=None, quantum_retained_variance=False, eps=0, theta=0, eta=0, theta_estimate=False,
                      use_computed_qcomponents=False,
                      eps_theta=0, ret_var=0, estimate_all=False, delta=0, error=False):

        U, S, Vt = self._fit(X, quantum_retained_variance=quantum_retained_variance, eps=eps, theta=theta,
                             eta=eta, theta_estimate=theta_estimate, use_computed_qcomponents=use_computed_qcomponents,
                             eps_theta=eps_theta, ret_var=ret_var, estimate_all=estimate_all, delta=delta)
        U = U[:, :self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0] - 1)
        else:
            # X_new = X * V = U * S * Vt * V = U * S
            U *= S[:self.n_components_]

        return U / self.spectral_norm

    def _fit(self, X, quantum_retained_variance, eps, theta_major, theta_minor, eta,
             theta_estimate,
             use_computed_qcomponents,
             eps_theta, ret_var, estimate_all, delta, true_tomography,
             fs_ratio_estimation, norm,
             stop_when_reached_accuracy, incremental_measure, faster_measure_increment, check_sv_uniform_distribution,
             spectral_norm_est, condition_number_est, estimate_least_k):
        """Dispatch to the right submethod depending on the chosen solver."""
        self.delta = delta
        self.eps_theta = eps_theta
        # self.classic_ret_variance_components = classic_ret_variance_components
        self.eps = eps
        self.theta_estimate = theta_estimate
        self.estimate_all = estimate_all
        self.estimate_least_k = estimate_least_k
        self.fs_ratio_estimation = fs_ratio_estimation
        self.quantum_retained_variance = quantum_retained_variance
        self.eta = eta
        # self.gamma = gamma
        self.theta_major = theta_major
        self.theta_minor = theta_minor
        self.ret_var = ret_var
        self.tomography_norm = norm
        self.true_tomography = true_tomography
        self.stop_when_reached_accuracy = stop_when_reached_accuracy
        self.incremental_measure = incremental_measure
        self.faster_measure_increment = faster_measure_increment
        self.check_sv_uniform_distribution = check_sv_uniform_distribution
        self.spectral_norm_est = spectral_norm_est
        self.condition_number_est = condition_number_est
        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        X = self._validate_data(X, dtype=[np.float64, np.float32],
                                ensure_2d=True, copy=self.copy)

        # Handle n_components==None
        if self.n_components is None:
            self.n_components_flag = False
            if self.svd_solver != 'arpack':
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            self.n_components_flag = True
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == 'auto':
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == 'mle':
                self._fit_svd_solver = 'full'
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                self._fit_svd_solver = 'randomized'
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = 'full'

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == 'full':
            return self._fit_full(X, n_components=n_components)
        elif self._fit_svd_solver in ['arpack', 'randomized']:
            warnings.warn('Attention! This computational path is purely classic!')
            return self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'"
                             "".format(self._fit_svd_solver))

    def _fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X."""
        n_samples, n_features = X.shape
        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='full'"
                             % (n_components, min(n_samples, n_features)))
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError("n_components=%r must be of type int "
                                 "when greater than or equal to 1, "
                                 "was of type=%r"
                                 % (n_components, type(n_components)))

        # Center data
        # self.spect_norm = max([np.linalg.norm(np.dot(X, x)) / np.linalg.norm(x) for x in X])
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt
        ##
        left_sv = U
        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.
        # Postprocess the number of components required
        if n_components == 'mle':
            n_components = \
                _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components,
                                           side='right') + 1
            # print(n_components)
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features

        if isinstance(self.ret_var, int):
            self.ret_var = np.sum(explained_variance_ratio_[:self.ret_var])

        if self.n_components_flag == False:
            n_components = self.ret_variance(explained_variance_ratio_, self.ret_var)
            self.components_retained_ = n_components
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.all_components=components_
        self.explained_variance_all=explained_variance_
        self.explained_variance_ratio_all=explained_variance_ratio_
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]
        self.left_sv = left_sv[:n_components]

        # Scaled singular values for assumptions purposes
        self.spectral_norm = self.singular_values_[0]
        self.frob_norm = np.linalg.norm(X)
        self.norm_muA, self.muA = best_mu(matrix=X, start=0, step=0.1)
        # self.quantum_components = self.q_ret_variance(1000, self.ret_var)

        if self.condition_number_est:
            self.est_cond_number = self.condition_number_estimation(epsilon=self.eps, delta=self.delta)

        if self.spectral_norm_est:
            self.est_spectral_norm = self.spectral_norm_estimation(epsilon=self.eps, delta=self.delta)

        '''if self.fs_ratio_estimation:
            self.estimate_fs_ratio, self.estimate_fs, self.estimate_s_values = self.quantum_factor_score_ratio_estimation(
                X, self.gamma, self.eps)'''

        if self.theta_estimate:
            self.est_theta = self.estimate_theta(epsilon=self.eps_theta, eta=self.eta, p=self.ret_var)

        if self.quantum_retained_variance:
            self.p = self.quantum_factor_score_ratio_sum(eps=self.eps, theta=self.theta_major, eta=self.eta)

        if self.estimate_least_k:
            self.estimate_least_right_sv, self.estimate_least_left_sv, self.estimate_least_s_values, self.estimate_least_fs, \
            self.estimate_least_fs_ratio = \
                self.least_k_sv_extractors(X=X, delta=self.delta, eps=self.eps, theta=self.theta_minor,
                                           true_tomography=self.true_tomography,
                                           norm=self.tomography_norm,
                                           stop_when_reached_accuracy=self.stop_when_reached_accuracy,
                                           incremental_measure=self.incremental_measure,
                                           faster_measure_increment=self.faster_measure_increment,
                                           check_sv_uniform_distribution=self.check_sv_uniform_distribution)

        if self.estimate_all:
            self.estimate_right_sv, self.estimate_left_sv, self.estimate_s_values, self.estimate_fs, self.estimate_fs_ratio = \
                self.topk_sv_extractors(X=X, delta=self.delta, eps=self.eps, theta=self.theta_major,
                                        true_tomography=self.true_tomography,
                                        norm=self.tomography_norm,
                                        stop_when_reached_accuracy=self.stop_when_reached_accuracy,
                                        incremental_measure=self.incremental_measure,
                                        faster_measure_increment=self.faster_measure_increment,
                                        check_sv_uniform_distribution=self.check_sv_uniform_distribution)

        return U, S, Vt

    def _fit_truncated(self, X, n_components, svd_solver):
        """Fit the model by computing truncated SVD (by ARPACK or randomized)
        on X.
        """
        n_samples, n_features = X.shape

        if isinstance(n_components, str):
            raise ValueError("n_components=%r cannot be a string "
                             "with svd_solver='%s'"
                             % (n_components, svd_solver))
        elif not 1 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 1 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='%s'"
                             % (n_components, min(n_samples, n_features),
                                svd_solver))
        elif not isinstance(n_components, numbers.Integral):
            raise ValueError("n_components=%r must be of type int "
                             "when greater than or equal to 1, was of type=%r"
                             % (n_components, type(n_components)))
        elif svd_solver == 'arpack' and n_components == min(n_samples,
                                                            n_features):
            raise ValueError("n_components=%r must be strictly less than "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='%s'"
                             % (n_components, min(n_samples, n_features),
                                svd_solver))

        random_state = check_random_state(self.random_state)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        if svd_solver == 'arpack':
            v0 = _init_arpack_v0(min(X.shape), random_state)
            U, S, Vt = svds(X, k=n_components, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            S = S[::-1]
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U[:, ::-1], Vt[::-1])

        elif svd_solver == 'randomized':
            # sign flipping is done inside
            U, S, Vt = randomized_svd(X, n_components=n_components,
                                      n_iter=self.iterated_power,
                                      flip_sign=True,
                                      random_state=random_state)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = Vt
        self.left_sv = U
        self.n_components_ = n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self.explained_variance_ratio_ = \
            self.explained_variance_ / total_var.sum()
        self.singular_values_ = S.copy()  # Store the singular values.

        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() -
                                    self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - n_components
        else:
            self.noise_variance_ = 0.

        # Scaled singular values for assumptions purposes
        self.spectral_norm = self.singular_values_[0]
        self.scaled_singular_values = self.singular_values_ / self.spectral_norm

        return U, S, Vt

    def score_samples(self, X):

        check_is_fitted(self)

        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision()
        log_like = -.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= .5 * (n_features * log(2. * np.pi) -
                          fast_logdet(precision))
        return log_like

    def score(self, X, y=None):

        return np.mean(self.score_samples(X))

    def _more_tags(self):
        return {'preserves_dtype': [np.float64, np.float32]}

    def transform(self, X, classic_transform=True, epsilon_delta=0,
                  quantum_representation=False, norm='None', psi=0, true_tomography=True,
                  use_classical_components=True):

        """Fit the model with X.

               Parameters
               ----------
               X : array-like of shape (n_samples, n_features)
                   Training data, where n_samples is the number of samples
                   and n_features is the number of features.

               classic_transform: bool, default=True.
                    If true, the classic transform is applied, otherwise quantum transform is applied.

               epsilon_delta: float, default=0.
                    Error to estimate the matrix (np.sqrt(n_components)*epsilon_delta)

               quantum_representation: bool, default=False.
                    If true it returns a different quantum representation of the data X, depending on the norm flag.

               norm : {'est_representation', 'q_state', 'None', 'f_norm'}, default='None'
                     If est_representation :
                        Estimates the U*Sigma matrix (lemma 13 QADRA)

                     If q_state :
                        Create and return the quantum state.

                     If None :
                        return only the estimate matrix with psi error

                     If f_norm :
                        return only the estimate matrix (with psi error) divided by its f-norm

               use_classical_components: bool, default=True.
                        If True, and classical_transform is False, it computes the classic transformation of the matrix X
                        and then an error is applied to this new matrix. If False, it computes the trasformation using the
                        estimated components with a certain error computed using Theorem 11 of QADRA.
               true_tomography: bool, default=True.
                        If true means that the quantum estimations are done with real tomography,
                        otherwise the estimations are approximated with a Truncated Gaussian Noise.

               Returns
               -------
               If classic_transform:
                        Returns the transformed matrix,
                        otherwise return a dictionary with all the result based on what you want to estimate.

               """

        if classic_transform:
            if epsilon_delta != 0 or quantum_representation or norm or psi != 0:
                warnings.warn("Warning! You are using the classical transform, so the quantum parameter are useless.")
            return super().transform(X)

        else:
            dict_res = {}
            X_final = super().transform(X, use_classical_components)
            # X_final = X_ / self.spectral_norm

            if use_classical_components == False:
                return X_final
            else:
                if quantum_representation:
                    assert (psi > 0 if norm != 'est_representation' else psi >= 0)
                    assert (epsilon_delta > 0)
                    result = self.compute_quantum_representation(X_final, psi=psi, epsilon_delta=epsilon_delta,
                                                                 type=norm, true_tomography=true_tomography)
                    dict_res.update({'quantum_representation_results': result})

                    return dict_res

    def inverse_transform(self, X, use_classical_components=True):
        return super().inverse_transform(X, use_classical_components)

    def compute_error(self, U, epsilon_delta, true_tomography):
        # error=epsilon+delta

        if true_tomography == False:
            epsilon_delta = np.sqrt(self.n_components_) * (epsilon_delta)

        A_sign = tomography(U, epsilon_delta, true_tomography=true_tomography)

        f_norm = np.linalg.norm(U - A_sign)
        return A_sign, epsilon_delta, f_norm

    def compute_quantum_representation(self, X, psi, epsilon_delta, true_tomography, type='None'):
        if type == 'est_representation':
            A_sign, epsilon_delta, f_norm = self.compute_error(X, epsilon_delta, true_tomography)
            return A_sign, epsilon_delta, f_norm
        elif type == 'q_state':
            Y_sign = tomography(X, psi, true_tomography=true_tomography)
            f_norm = np.linalg.norm(Y_sign)
            Yi_ = []
            norm_Y = []

            for i in range(len(Y_sign)):
                norm_Y.append(np.linalg.norm(Y_sign[i, :], ord=2) / f_norm)
                Yi_.append(Y_sign[i, :] / f_norm)
            q_state = QuantumState(registers=Yi_, amplitudes=norm_Y)
            return q_state
        elif type == 'None':
            Y_sign = tomography(X, psi, true_tomography=true_tomography)
            return Y_sign
        elif type == 'f_norm':
            Y_sign = tomography(X, psi, true_tomography=true_tomography)
            f_norm = np.linalg.norm(Y_sign)
            return Y_sign / f_norm

    def spectral_norm_estimation(self, epsilon, delta):
        l = 0
        u = 1
        n_iterations = int(np.ceil((np.log(self.frob_norm / epsilon))))
        tau = (l + u) / 2
        for i in range(n_iterations):

            theta_from_sv = np.array(
                [wrapper_phase_est_arguments(sv) / ((1 / epsilon) + np.pi) for sv in
                 self.singular_values_ / self.frob_norm])
            theta_estimations = [consistent_phase_estimation(omega=theta_, epsilon=epsilon / self.frob_norm,
                                                             gamma=1 - 1 / self.n_features_) for
                                 theta_ in
                                 theta_from_sv]
            est_sing_values = np.array([unwrap_phase_est_arguments(th, eps=1 / epsilon) for th in theta_estimations])
            selected_sing_values = self.singular_values_[est_sing_values >= tau]
            eta = np.sum(np.square(selected_sing_values)) / (self.frob_norm ** 2)
            eta_est = amplitude_estimation(a=eta, epsilon=delta)

            if eta_est == 0:
                u = tau
            else:
                l = tau
            tau = (u + l) / 2

        return tau * self.frob_norm

    def condition_number_estimation(self, epsilon, delta):
        '''l = 0
        u = 1
        n_iterations = int(np.ceil((np.log(self.frob_norm / epsilon))))
        tau = (l + u) / 2
        for i in range(n_iterations):
            theta_from_sv = np.array(
                [wrapper_phase_est_arguments(sv) / ((1 / epsilon) + np.pi) for sv in
                 self.singular_values_ / self.frob_norm])
            theta_estimations = [consistent_phase_estimation(omega=theta_, epsilon=epsilon / self.frob_norm,
                                                             delta=1 - 1 / self.n_features_) for
                                 theta_ in
                                 theta_from_sv]
            est_sing_values = np.array([unwrap_phase_est_arguments(th, eps=1 / epsilon) for th in theta_estimations])
            selected_sing_values = self.singular_values_[est_sing_values <= tau]
            eta = np.sum(np.square(selected_sing_values)) / (self.frob_norm ** 2)
            if eta > 1:
                eta = 1
            sing_values_reversed_squared = (self.singular_values_[::-1] ** 2) / (self.frob_norm ** 2)
            eta_est = amplitude_estimation(theta=eta, epsilon=delta)
            k_first = self.singular_values_[::-1][np.where(np.cumsum(sing_values_reversed_squared) >= eta_est)[0][0]]
            if eta_est == 1:
                u = tau
            else:
                l = tau
            tau = (u + l) / 2
        sing_min = self.spectral_norm / k_first'''
        l = 0
        u = 1
        n_iterations = int(np.ceil((np.log(self.frob_norm / epsilon))))
        tau = (l + u) / 2
        for i in range(n_iterations):
            theta_from_sv = np.array(
                [wrapper_phase_est_arguments(sv) / ((1 / epsilon) + np.pi) for sv in
                 self.singular_values_ / self.frob_norm])
            theta_estimations = [consistent_phase_estimation(omega=theta_, epsilon=epsilon / self.frob_norm,
                                                             delta=1 - 1 / self.n_features_) for
                                 theta_ in
                                 theta_from_sv]
            est_sing_values = np.array([unwrap_phase_est_arguments(th, eps=1 / epsilon) for th in theta_estimations])
            selected_sing_values = self.singular_values_[est_sing_values <= tau]
            eta = np.sum(np.square(selected_sing_values)) / (self.frob_norm ** 2)
            if eta > 1:
                eta = 1
            eta_est = amplitude_estimation(theta=eta, epsilon=delta)

            if eta_est == 1:
                u = tau
            else:
                l = tau
            tau = (u + l) / 2
        sing_min = self.singular_values_[0] / (tau * self.frob_norm)
        return tau * self.frob_norm

    # Theorem 8 New Qadra
    '''def quantum_factor_score_ratio_estimation(self, X, gamma, epsilon):

        singular_values = self.singular_values_[self.explained_variance_ratio_ > gamma] / self.muA
        theta_from_sv = np.array([wrapper_phase_est_arguments(sv) / np.pi for sv in singular_values])
        theta_estimations = [
            consistent_phase_estimation(epsilon / self.muA, gamma=1 - 1 / self.n_features_, omega=theta) for theta in
            theta_from_sv]
        singular_values_est = np.array([unwrap_phase_est_arguments(th) for th in theta_estimations])

        factor_score_est = singular_values_est ** 2
        factor_score_ratio_est = np.array([fs / (np.linalg.norm(X) ** 2) for fs in factor_score_est])
        q = QuantumState(registers=singular_values_est, amplitudes=factor_score_ratio_est)
        #cc = coupon_collect(q)
        return factor_score_ratio_est, (
                singular_values_est / self.spectral_norm) ** 2, singular_values_est / self.spectral_norm
'''

    # Theorem 9
    def quantum_factor_score_ratio_sum(self, eps, theta, eta):
        if theta:
            pass
        else:
            theta = self.est_theta
        theta_from_sv = np.array(
            [wrapper_phase_est_arguments(sv) / (eps + np.pi) for sv in self.singular_values_ / self.muA])
        theta_estimations = [consistent_phase_estimation(omega=theta_, epsilon=eps, gamma=1 - 1 / self.n_features_) for
                             theta_ in theta_from_sv]
        est_selected_sing_values = np.array([unwrap_phase_est_arguments(th, eps=eps) for th in theta_estimations])
        selected_sing_values = self.singular_values_[est_selected_sing_values >= theta]

        pow2 = lambda x: x ** 2
        selected_sing_values_squared = np.apply_along_axis(pow2, 0, selected_sing_values)
        sing_values_squared = np.apply_along_axis(pow2, 0, self.singular_values_)
        p = sum(selected_sing_values_squared) / sum(sing_values_squared)
        p_est = amplitude_estimation(a=p, epsilon=eta)
        return p_est

    # new Theorem 10
    def estimate_theta(self, epsilon, eta, p):
        l = 0
        u = 1
        print('Ok_theta')
        if np.abs(l - p) <= eta:
            return self.muA
        if np.abs(u - p) <= eta:
            return 0
        n_iterations = int(np.ceil((np.log(self.muA / epsilon))))
        tau = (l + u) / 2
        for i in range(n_iterations):
            p_est = self.quantum_factor_score_ratio_sum(eps=epsilon / self.muA, theta=tau, eta=eta / 2)
            print(tau, p_est - p)
            if np.abs(p_est - p) <= eta / 2:
                return tau * self.muA
            if p_est < p:
                u = tau
            else:
                l = tau
            tau = (u + l) / 2
        raise ValueError("The binary search doesn't found any values")

    # Theorem 11
    def topk_sv_extractors(self, X, delta, eps, theta, true_tomography, norm, stop_when_reached_accuracy,
                           incremental_measure, faster_measure_increment, check_sv_uniform_distribution):

        if theta == 0:
            theta = self.est_theta

        theta_from_sv = np.array(
            [wrapper_phase_est_arguments(sv) / ((eps / self.muA) + np.pi) for sv in self.singular_values_ / self.muA])
        theta_estimations = [
            consistent_phase_estimation(omega=theta_, epsilon=eps / self.muA, gamma=1 - 1 / self.n_features_) for
            theta_ in theta_from_sv]
        est_selected_sing_values = np.array(
            [unwrap_phase_est_arguments(th, eps=(eps / self.muA)) * self.muA for th in theta_estimations])
        self.top_k_true_singular_value = self.singular_values_[est_selected_sing_values >= theta]
        singular_value_estimation = est_selected_sing_values[est_selected_sing_values >= theta]

        if check_sv_uniform_distribution:
            distribution_k = self.top_k_true_singular_value / est_selected_sing_values
            plt.plot(distribution_k)
            plt.show()
        self.topk = len(singular_value_estimation)
        pow2 = lambda x: x ** 2
        selected_sing_values_squared = np.apply_along_axis(pow2, 0, self.top_k_true_singular_value)
        sing_values_squared = np.apply_along_axis(pow2, 0, self.singular_values_)
        self.topk_p = sum(selected_sing_values_squared) / sum(sing_values_squared)
        self.topk_right_singular_vectors = self.components_[est_selected_sing_values >= theta]
        self.topk_left_singular_vectors = self.left_sv[est_selected_sing_values >= theta]

        right_singular_vectors_est = tomography(self.topk_right_singular_vectors, delta,
                                                true_tomography=true_tomography,
                                                norm=norm,
                                                stop_when_reached_accuracy=stop_when_reached_accuracy,
                                                incremental_measure=incremental_measure,
                                                faster_measure_increment=faster_measure_increment)
        left_singular_vectors_est = tomography(self.topk_left_singular_vectors, delta, true_tomography=true_tomography,
                                               norm=norm,
                                               stop_when_reached_accuracy=stop_when_reached_accuracy,
                                               incremental_measure=incremental_measure,
                                               faster_measure_increment=faster_measure_increment)
        factor_score_estimation = singular_value_estimation ** 2
        factor_score_ratio_est = np.array([fs / (np.linalg.norm(X) ** 2) for fs in factor_score_estimation])

        return right_singular_vectors_est, left_singular_vectors_est, singular_value_estimation, \
               (singular_value_estimation ** 2) / (self.n_samples_ - 1), factor_score_ratio_est

    def least_k_sv_extractors(self, X, delta, eps, theta, true_tomography, norm, stop_when_reached_accuracy,
                              incremental_measure, faster_measure_increment, check_sv_uniform_distribution):

        if theta == 0:
            theta = self.least_theta

        theta_from_sv = np.array(
            [wrapper_phase_est_arguments(sv) / ((eps / self.muA) + np.pi) for sv in
             self.singular_values_[:np.where(np.isclose(self.singular_values_, 0))[0][0]] / self.muA])
        theta_estimations = [
            consistent_phase_estimation(omega=theta_, epsilon=eps / self.muA, gamma=1 - 1 / self.n_features_) for
            theta_ in
            theta_from_sv]
        est_selected_sing_values = np.array(
            [unwrap_phase_est_arguments(th, eps=(eps / self.muA)) * self.muA for th in theta_estimations])
        self.least_k_true_singular_value = self.singular_values_[:np.where(np.isclose(self.singular_values_, 0))[0][0]][
            est_selected_sing_values < theta]
        singular_value_estimation = est_selected_sing_values[est_selected_sing_values < theta]

        if check_sv_uniform_distribution:
            distribution_k = self.least_k_true_singular_value / est_selected_sing_values
            plt.plot(distribution_k)
            plt.show()

        self.least_k = len(singular_value_estimation)

        pow2 = lambda x: x ** 2
        selected_sing_values_squared = np.apply_along_axis(pow2, 0, self.least_k_true_singular_value)
        sing_values_squared = np.apply_along_axis(pow2, 0, self.singular_values_)
        self.least_k_p = sum(selected_sing_values_squared) / sum(sing_values_squared)
        self.leastk_right_singular_vectors = self.components_[:np.where(np.isclose(self.singular_values_, 0))[0][0]][
            est_selected_sing_values < theta]
        self.leastk_left_singular_vectors = self.left_sv[:np.where(np.isclose(self.singular_values_, 0))[0][0]][
            est_selected_sing_values < theta]

        right_singular_vectors_est = tomography(self.leastk_right_singular_vectors, delta,
                                                true_tomography=true_tomography,
                                                norm=norm,
                                                stop_when_reached_accuracy=stop_when_reached_accuracy,
                                                incremental_measure=incremental_measure,
                                                faster_measure_increment=faster_measure_increment)
        left_singular_vectors_est = tomography(self.leastk_left_singular_vectors, delta,
                                               true_tomography=true_tomography,
                                               norm=norm,
                                               stop_when_reached_accuracy=stop_when_reached_accuracy,
                                               incremental_measure=incremental_measure,
                                               faster_measure_increment=faster_measure_increment)
        factor_score_estimation = singular_value_estimation ** 2
        factor_score_ratio_est = np.array([fs / (np.linalg.norm(X) ** 2) for fs in factor_score_estimation])

        return right_singular_vectors_est, left_singular_vectors_est, singular_value_estimation, \
               (singular_value_estimation ** 2) / (self.n_samples_ - 1), factor_score_ratio_est

    def accumulate_q_runtime(self, n_samples, n_features, estimate_components='all'):
        if self.theta_major == 0:
            self.theta = self.est_theta
        if self.theta_estimate:
            self.quantum_runtime_container.append(
                (self.muA * np.log(self.muA / self.eps_theta) * np.log(n_samples * n_features)) / (
                        self.eps_theta * self.eta))
        if self.quantum_retained_variance:
            self.quantum_runtime_container.append(self.muA / (self.eps * self.eta))
        if self.estimate_all:
            if self.tomography_norm == 'L2':
                cost_left_sv = (self.spectral_norm * self.muA * self.topk * np.log(self.topk) *
                                n_samples * np.log(n_samples)) / (
                                       self.theta * np.sqrt(self.topk_p) * self.eps * self.delta ** 2)
                cost_right_sv = ((self.spectral_norm / self.theta) * (1 / np.sqrt(self.topk_p)) * (
                        self.muA / self.eps) * ((self.topk * np.log(self.topk) *
                                                 n_features * np.log(n_features)) / (self.delta ** 2)))
            else:
                cost_left_sv = np.full(shape=n_samples.shape,
                                       fill_value=((self.spectral_norm * self.muA * self.topk) / (
                                               self.theta * self.eps * self.delta ** 2)))

                cost_right_sv = np.full(shape=n_features.shape,
                                        fill_value=((self.spectral_norm * self.muA * self.topk) / (
                                                self.theta * self.eps * self.delta ** 2)))

            if estimate_components == 'all':
                self.quantum_runtime_container.append(cost_left_sv + cost_right_sv +
                                                      ((self.spectral_norm * self.muA * self.topk * np.log(
                                                          self.topk)) /
                                                       (self.theta * np.sqrt(self.topk_p) * self.eps)))
            elif estimate_components == 'left_sv':
                self.quantum_runtime_container.append(cost_left_sv +
                                                      ((self.spectral_norm * self.muA * self.topk * np.log(
                                                          self.topk)) /
                                                       (self.theta * np.sqrt(self.topk_p) * self.eps)))
            elif estimate_components == 'right_sv':
                self.quantum_runtime_container.append(cost_right_sv +
                                                      ((self.spectral_norm * self.muA * self.topk * np.log(self.topk)) /
                                                       (self.theta * np.sqrt(self.topk_p) * self.eps)))
        # TODO: verificare costo di theta e spectral norm
        if self.estimate_least_k:
            if self.tomography_norm == 'L2':
                cost_least_left_sv = ((self.theta_minor /
                                       self.singular_values_[:(np.where(np.isclose(self.singular_values_, 0)))[0][0]][
                                           -1])
                                      * (1 / np.sqrt(self.least_k_p)) * (self.muA / self.eps) *
                                      ((self.least_k * np.log(self.least_k) * n_samples * np.log(n_samples)) / (
                                              self.delta ** 2)))

                cost_least_right_sv = ((self.theta_minor /
                                        self.singular_values_[:(np.where(np.isclose(self.singular_values_, 0)))[0][0]][
                                            -2])
                                       * (1 / np.sqrt(self.least_k_p)) * (self.muA / self.eps) *
                                       ((self.least_k * np.log(self.least_k) * n_features * np.log(n_features)) / (
                                               self.delta ** 2)))
            else:
                cost_least_left_sv = np.full(shape=n_samples.shape,
                                             fill_value=((self.spectral_norm * self.muA * self.least_k) / (
                                                     self.theta_minor * self.eps * self.delta ** 2)))

                cost_least_right_sv = np.full(shape=n_features.shape,
                                              fill_value=((self.spectral_norm * self.muA * self.least_k) / (
                                                      self.theta_minor * self.eps * self.delta ** 2)))

            if estimate_components == 'all':
                self.quantum_runtime_container.append(cost_least_right_sv + cost_least_left_sv +
                                                      ((self.theta_minor * self.muA * self.least_k) /
                                                       (self.singular_values_[
                                                        :(np.where(np.isclose(self.singular_values_, 0)))[0][0]][-2]
                                                        * np.sqrt(self.least_k_p) * self.eps)))
            elif estimate_components == 'left_sv':
                self.quantum_runtime_container.append(cost_least_left_sv +
                                                      ((self.theta_minor * self.muA * self.least_k) /
                                                       (self.singular_values_[
                                                        :(np.where(np.isclose(self.singular_values_, 0)))[0][0]][-1]
                                                        * np.sqrt(self.least_k_p) * self.eps)))
            elif estimate_components == 'right_sv':
                self.quantum_runtime_container.append(cost_least_right_sv +
                                                      ((self.theta_minor * self.muA * self.least_k) /
                                                       (self.singular_values_[
                                                        :(np.where(np.isclose(self.singular_values_, 0)))[0][0]][-2]
                                                        * np.sqrt(self.least_k_p) * self.eps)))
        '''if self.fs_ratio_estimation:
            self.quantum_runtime_container.append(self.muA / (self.eps * self.gamma ** 2))'''
        return self.quantum_runtime_container

    def q_ret_variance(self, measurements, variance):
        if isinstance(self.n_components, int):
            return self.n_components
        else:
            q_state = QuantumState(registers=self.scaled_singular_values, amplitudes=self.scaled_singular_values)
            estimations = estimate_wald(q_state.measure(measurements))
            exp_var = 0
            i = 0
            sv = sorted(estimations.keys(), reverse=True)
            dict_elem = np.array(list(map(estimations.get, sv)))
            exp_var = 0
            i = 0
            while exp_var <= variance:
                exp_var += dict_elem[i]
                i += 1
            k = i
            return k

    def ret_variance(self, explained_variance_ratio_, variance):
        ratio_cumsum = stable_cumsum(explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, variance,
                                       side='right') + 1

        return n_components

    def runtime_comparison(self, n_samples, n_features, saveas, estimate_components='all', classic_runtime='classic'):
        """Function that allows to compare classic vs quantum runtime of the algorithm executed.

        Parameters
        ----------
        n_samples: int value.
            The number of samples that you want to simulate in the runtime measurements.

        n_features: int value.
            The number of features that you want to simulate in the runtime measurements.

        saveas: string value.
            Name under which the image will be saved. Specify also the format, otherwise the image will be saved with
            the .fig MATLAB format.

        estimate_components: string value, default='all'.
            If 'all':
                It means that you have estimated the left/right singular vectors, the singular values, factor score and
                factor score ratios.
            If 'left_sv':
                It means that you have estimated the left singular vectors, the singular values, factor score and
                factor score ratios.
            If 'right_sv':
                It means that you have estimated the right singular vectors, the singular values, factor score and
                factor score ratios.

        classic_runtime: string value, default='classic'.
            If 'classic':
                It means that you want compare the quantum runtime with the classic runtime version of PCA, that is
                O(nm^2) where n are the number of samples and m the number of features.
            If 'rand':
                It means that you want compare the quantum runtime with the randomized runtime version of PCA, that is
                O(nmk*log(m*eps/eps)) where k are the components retained and eps is an approximation error related to
                the relative spectral gap between eigenvalues.

        Returns
        -------
        This functions doesn't return anything, it just save the runtime comparison plot.

        Notes
        -------
        This function use the MATLAB engine. This because the MATLAB plots are more visible and clear.
        """

        n, m = np.meshgrid(np.linspace(1, n_samples, dtype=np.int64, num=100),
                           np.linspace(1, n_features, dtype=np.int64, num=100))

        if classic_runtime == 'rand':
            '''eps = self.explained_variance_[0] - self.explained_variance_[
                1]
            c_runtime = n * m * self.components_retained_ * np.log(
                m / eps) / np.sqrt(eps)'''
            # c_runtime = n*m*np.log(self.components_retained_)+(m+n)*self.components_retained_**2
            c_runtime = (n * m * np.log(self.components_retained_))
        elif classic_runtime == 'classic':
            c_runtime = n * m ** 2

        q_runtime = self.accumulate_q_runtime(n_samples=n, n_features=m,
                                              estimate_components=estimate_components)
        if len(q_runtime) > 1:
            q_runtime = np.sum(q_runtime, axis=0)
        else:
            q_runtime = q_runtime[0]

        eng = matlab.engine.start_matlab()
        c_runtime_matlab = matlab.double(c_runtime.tolist())
        q_runtime_matlab = matlab.double(q_runtime.tolist())
        n_matlab = matlab.double(n.tolist())
        m_matlab = matlab.double(m.tolist())

        fig = eng.figure()
        eng.plot3(n_matlab, m_matlab, q_runtime_matlab, '-b', 'DisplayName', 'quantumRuntime', nargout=0)
        eng.hold("on", nargout=0)
        eng.plot3(n_matlab, m_matlab, c_runtime_matlab, '-g', 'DisplayName', 'classicRuntime', nargout=0)
        eng.hold("off", nargout=0)
        # eng.legend('{\color{green}classicRuntime}', '{\color{blue}quantumRuntime}', nargout=0)
        eng.ylabel('nFeatures', nargout=0)
        eng.xlabel('nSamples', nargout=0)
        eng.title(self.name + ' VS ' + 'q-' + self.name, nargout=0)
        eng.saveas(fig, saveas, nargout=0)
        eng.quit()
