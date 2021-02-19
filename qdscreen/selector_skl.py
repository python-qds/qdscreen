import numpy as np

from sklearn.exceptions import NotFittedError

from .compat import BaseEstimator, SelectorMixin
from .main import qd_screen


class QDSSelector(SelectorMixin, BaseEstimator):
    """Feature selector that removes all features that are (quasi-)deterministically predicted from others.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the User Guide.

    Parameters
    ----------
    absolute_eps : float, optional
        Absolute entropy threshold. Any feature Y that can be predicted from
        another feature X in a quasi-deterministic way, that is, where
        conditional entropy H(Y|X) <= absolute_eps, will be removed. The default
        value is 0 and corresponds to removing deterministic relationships only.
    relative_eps : float, optional
        Relative entropy threshold. Any feature Y that can be predicted from
        another feature X in a quasi-deterministic way, that is, where relative
        conditional entropy H(Y|X)/H(Y) <= relative_eps (between 0 and 1), will
        be removed. Only one of absolute_eps and relative_eps should be
        provided.

    Attributes
    ----------
    model_ : instance of ``QDForest``
        Variances of individual features.

    Notes
    -----
    Allows NaN in the input.

    Examples
    --------
    TODO make this better ? see test_readme.py
    The following dataset has integer features, two of which are constant, and all of which being 'predictable' from
    the third one::

    >>> X = [[0, 2, 0, 3],
    ...      [0, 1, 4, 3],
    ...      [0, 1, 1, 3]]
    >>> selector = QDSSelector()
    >>> Xsel = selector.fit_transform(X)
    >>> Xsel
    array([[0],
           [4],
           [1]])
    >>> selector.inverse_transform(Xsel)
    array([[0, 2, 0, 3],
    ...    [0, 1, 4, 3],
    ...    [0, 1, 1, 3]])
    """

    def __init__(self,
                 absolute_eps=None, relative_eps=None):
        self.absolute_eps = absolute_eps
        self.relative_eps = relative_eps

    def fit(self, X, y=None):
        """Learn determinism from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        X = self._validate_data(X, accept_sparse=False,  #('csr', 'csc'),
                                dtype=np.object,
                                force_all_finite='allow-nan')

        # if hasattr(X, "toarray"):   # sparse matrix
        #     _, self.variances_ = mean_variance_axis(X, axis=0)
        #     if self.threshold == 0:
        #         mins, maxes = min_max_axis(X, axis=0)
        #         peak_to_peaks = maxes - mins
        # else:

        # First find the forest structure
        forest_ = qd_screen(X, absolute_eps=self.absolute_eps, relative_eps=self.relative_eps)

        # Then learn the parameter maps
        self.model_ = forest_.fit_selector_model(X)

        return self

    def check_is_fitted(self):
        if not hasattr(self, "model_"):
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg % {'name': type(self).__name__})

    def _get_support_mask(self):
        self.check_is_fitted()
        return self.model_.forest.roots_mask_ar

    def inverse_transform(self, X):
        """
        Reverse the transformation operation

        Parameters
        ----------
        X : array of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_original_features]
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        Xt = super(QDSSelector, self).inverse_transform(X)
        # use inplace = True because Xt is already prepared
        self.model_.predict_qd(Xt, inplace=True)
        return Xt

    def _more_tags(self):
        return {'allow_nan': True}
