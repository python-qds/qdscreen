import numpy as np
import pandas as pd
from scipy import sparse

try:
    from typing import Union, Optional
except:  # noqa
    pass

from .compat import SelectorMixin, BaseEstimator, NotFittedError
from .main import qd_screen, QDForest


class QDSelectorModel(object):
    """
    A quasi-determinism feature selection model that can be

     - fit from a dataset using <model>.fit(X)
     - used to select only the relevant (root) features using <model>.remove_qd(X)
     - used to predict the other columns from the relevant (root) ones using <model>.predict_qd(X)

    """
    __slots__ = ('forest', # the QDForest
                 '_maps'   # a sparse square array (parent, child): mapping_dct with index in the order of self.varnames
                 )

    def __init__(self,
                 qd_forest  # type: QDForest
                 ):
        self.forest = qd_forest
        self._maps = None

    def fit(self,
            X  # type: Union[np.ndarray, pd.DataFrame]
            ):
        """Fits the maps able to predict determined features from others"""
        forest = self.forest

        # we will create a sparse coordinate representation of maps
        n = forest.nb_vars

        if forest.is_nparray:
            assert isinstance(X, np.ndarray)

            # detect numpy structured arrays
            is_struct_array = X.dtype.names is not None
            if is_struct_array:
                # names = X.dtype.names
                # assert len(names) == n
                raise NotImplementedError("structured numpy arrays are not supported. Please convert your array to an "
                                          "unstructured array")
            else:
                assert X.shape[1] == n

            # self._maps = maps = sparse.coo_matrix((n, n), dtype=object)  # not convenient for incremental
            self._maps = maps = sparse.dok_matrix((n, n), dtype=object)

            for parent, child in forest.get_arcs():
                # todo maybe remove this check later for efficiency
                assert (parent, child) not in maps, "Error: edge already exists"

                # create a dictionary mapping each parent level to most frequent child level
                # -- seems suboptimal with numpy...
                # map_dct = dict()
                # for parent_lev in np.unique(X[:, parent]):
                #     values, counts = np.unique(X[X[:, parent] == parent_lev, child], return_counts=True)
                #     map_dct[parent_lev] = values[np.argmax(counts)]
                # -- same with pandas groupby
                # if is_struct_array:
                #     pc_df = pd.DataFrame(X[[names[parent], names[child]]])
                #     pc_df.columns = [0, 1]  # forget the names
                # else:
                pc_df = pd.DataFrame(X[:, (parent, child)])
                levels_mapping_df = pc_df.groupby(0).agg(lambda x: x.value_counts().index[0])
                maps[parent, child] = levels_mapping_df.iloc[:, 0].to_dict()

        else:
            assert isinstance(X, pd.DataFrame)

            # unfortunately pandas dataframe sparse do not allow item assignment :( so we need to work on numpy array
            # first get the numpy array in correct order
            varnames = forest.varnames
            X_ar = X.loc[:, varnames].values
            self._maps = maps = sparse.dok_matrix((n, n), dtype=object)

            for parent, child in forest.get_arcs(names=False):
                # todo maybe remove this check later for efficiency
                assert (parent, child) not in maps, "Error: edge already exists"
                # levels_mapping_df = X.loc[:, (parent, child)].groupby(parent).agg(lambda x: x.value_counts().index[0])
                # maps[parent, child] = levels_mapping_df[child].to_dict()
                levels_mapping_df = pd.DataFrame(X_ar[:, (parent, child)]).groupby(0).agg(lambda x: x.value_counts().index[0])
                maps[parent, child] = levels_mapping_df.iloc[:, 0].to_dict()

    def remove_qd(self,
                  X,             # type: Union[np.ndarray, pd.DataFrame]
                  inplace=False  # type: bool
                  ):
        # type: (...) -> Optional[Union[np.ndarray, pd.DataFrame]]
        """
        Removes from X the features that can be (quasi-)determined from the others
        This returns a copy by default, except if `inplace=True`

        :param X:
        :param inplace: if this is set to True,
        :return:
        """
        forest = self.forest

        is_x_nparray = isinstance(X, np.ndarray)
        assert is_x_nparray == forest.is_nparray

        if is_x_nparray:
            is_structured = X.dtype.names is not None
            if is_structured:
                raise NotImplementedError("structured numpy arrays are not supported. Please convert your array to an "
                                          "unstructured array")
            if inplace:
                np.delete(X, forest.roots_mask_ar, axis=1)
            else:
                # for structured: return X[np.array(X.dtype.names)[forest.roots_mask_ar]]
                return X[:, forest.roots_mask_ar]
        else:
            # pandas dataframe
            if inplace:
                del X[forest.roots]
            else:
                return X.loc[:, forest.roots_mask]

    def predict_qd(self,
                   X,             # type: Union[np.ndarray, pd.DataFrame]
                   inplace=False  # type: bool
                   ):
        # type: (...) -> Optional[Union[np.ndarray, pd.DataFrame]]
        """
        Adds columns to X corresponding to the features that can be determined from the roots.
        By default,

        :param X:
        :param inplace: if `True` and X is a dataframe, predicted columns will be added inplace. Note that the order
            may differ from the initial trainin
        :return:
        """
        forest = self.forest

        # if inplace is None:
        #     inplace = not self.is_nparray

        is_x_nparray = isinstance(X, np.ndarray)
        assert is_x_nparray == forest.is_nparray

        if is_x_nparray:
            is_structured = X.dtype.names is not None
            if is_structured:
                raise NotImplementedError("structured numpy arrays are not supported. Please convert your array to an "
                                          "unstructured array")
            if not inplace:
                # Same as in sklearn inverse_transform: create the missing columns in X first
                X_in = X
                support = forest.roots_mask_ar
                # X = check_array(X, dtype=None)
                nbcols_received = X_in.shape[1]
                if support.sum() != nbcols_received:
                    raise ValueError("X has a different nb columns than the number of roots found during fitting.")
                # if a single column, make sure this is 2d
                if X_in.ndim == 1:
                    X_in = X_in[None, :]
                # create a copy with the extra columns
                X = np.zeros((X_in.shape[0], support.size), dtype=X_in.dtype)
                X[:, support] = X_in
            else:
                if X.shape[1] != forest.nb_vars:
                    raise ValueError("If `inplace=True`, `predict` expects an X input with the correct number of "
                                     "columns. Use `inplace=False` to pass only the array of roots. Note that this"
                                     "is the default behaviour of inplace.")

            # walk the tree from the roots
            for _, parent, child in forest.walk_arcs():
                # apply the learned map efficienty https://stackoverflow.com/q/16992713/7262247
                X[:, child] = np.vectorize(self._maps[parent, child].__getitem__)(X[:, parent])
        else:
            if not inplace:
                X = X.copy()

            # walk the tree from the roots
            varnames = forest.varnames
            for _, parent, child in forest.walk_arcs(names=False):
                # apply the learned map efficienty https://stackoverflow.com/q/16992713/7262247
                X.loc[:, varnames[child]] = np.vectorize(self._maps[parent, child].__getitem__)(X.loc[:, varnames[parent]])

        if not inplace:
            return X


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
    TODO variances_ : array, shape (n_features,)
        Variances of individual features.

    Notes
    -----
    Allows NaN in the input.

    Examples
    --------
    TODO The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = QDSSelector()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
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
