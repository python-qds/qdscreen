import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode

try:
    from typing import Union, Optional, Dict, Any
except:  # noqa
    pass

from .main import QDForest


class InvalidDataInputError(ValueError):
    """Raised when input data is invalid"""


def _get_most_common_value(x):
    # From https://stackoverflow.com/a/47778607/7262247
    # `scipy_mode` is the most robust to the various pitfalls (nans, ...)
    # but they will deprecate it
    # return scipy_mode(x, nan_policy=None)[0][0]
    res = x.mode(dropna=True)
    if len(res) == 0:
        return np.nan
    else:
        return res


class ParentChildMapping:
    __slots__ = ('_mapping_dct', '_otypes')

    def __init__(
        self,
        mapping_dct  # type: Dict
    ):
        self._mapping_dct = mapping_dct
        # Find the correct otype to use in the vectorized operation
        self._otypes = [np.array(mapping_dct.values()).dtype]

    def predict_child_from_parent_ar(
        self,
        parent_values  # type: np.ndarray
    ):
        """For numpy"""
        # apply the learned map efficienty https://stackoverflow.com/q/16992713/7262247
        return np.vectorize(self._mapping_dct.__getitem__, otypes=self._otypes)(parent_values)

    def predict_child_from_parent(
        self,
        parent_values  # type: pd.DataFrame
    ):
        """For pandas"""
        # See https://stackoverflow.com/questions/47930052/pandas-vectorized-lookup-of-dictionary
        return parent_values.map(self._mapping_dct)


class QDSelectorModel(object):
    """
    A quasi-determinism feature selection model that can be

     - fit from a dataset using <model>.fit(X)
     - used to select only the relevant (root) features using <model>.remove_qd(X)
     - used to predict the other columns from the relevant (root) ones using <model>.predict_qd(X)

    """
    __slots__ = ('forest', # the QDForest
                 '_maps'   # a nested dict {parent: {child: mapping_dct with index in the order of self.varnames}}
                           # note: scipy.sparse now raises an error with dtype=object
                 )

    def __init__(self,
                 qd_forest  # type: QDForest
                 ):
        self.forest = qd_forest
        self._maps = None  # type: Optional[Dict[Any, Dict[Any, Dict]]]

    def assert_valid_input(
        self,
        X,  # type: Union[np.ndarray, pd.DataFrame]
        df_extras_allowed=False  # type: bool
    ):
        """Raises an InvalidDataInputError if X does not match the expectation"""

        if self.forest.is_nparray:
            if not isinstance(X, np.ndarray):
                raise InvalidDataInputError(
                    "Input data must be an numpy array. Found: %s" % type(X))

            if X.shape[1] != self.forest.nb_vars:  # or X.shape[0] != X.shape[1]:
                raise InvalidDataInputError(
                    "Input numpy array must have %s columns. Found %s columns" % (self.forest.nb_vars, X.shape[1]))
        else:
            if not isinstance(X, pd.DataFrame):
                raise InvalidDataInputError(
                    "Input data must be a pandas DataFrame. Found: %s" % type(X))

            actual = set(X.columns)
            expected = set(self.forest.varnames)
            if actual != expected:
                missing = expected - actual
                if missing or not df_extras_allowed:
                    extra = actual - expected
                    raise InvalidDataInputError(
                        "Input pandas DataFrame must have column names matching the ones in the model. "
                        "Missing: %s. Extra: %s " % (missing, extra)
                    )

    def fit(
        self,
        X  # type: Union[np.ndarray, pd.DataFrame]
    ):
        """Fits the maps able to predict determined features from others"""
        forest = self.forest

        # Validate the input
        self.assert_valid_input(X, df_extras_allowed=False)

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

            self._maps = maps = dict()

            for parent, child in forest.get_arcs():
                # assert (parent, child) not in maps, "Error: edge already exists"

                # create a dictionary mapping each parent level to most frequent child level
                #
                # -- seems suboptimal with numpy...
                # map_dct = dict()
                # for parent_lev in np.unique(X[:, parent]):
                #     values, counts = np.unique(X[X[:, parent] == parent_lev, child], return_counts=True)
                #     map_dct[parent_lev] = values[np.argmax(counts)]
                #
                # -- same with pandas groupby
                # if is_struct_array:
                #     pc_df = pd.DataFrame(X[[names[parent], names[child]]])
                #     pc_df.columns = [0, 1]  # forget the names
                # else:
                pc_df = pd.DataFrame(X[:, (parent, child)], columns=["parent", "child"])
                levels_mapping_df = pc_df.groupby(by="parent").agg(_get_most_common_value)

                # Init the dict for parent if it does not exit
                maps.setdefault(parent, dict())

                # Fill the parent-child item with the mapping object
                maps[parent][child] = ParentChildMapping(levels_mapping_df.iloc[:, 0].to_dict())

        else:
            assert isinstance(X, pd.DataFrame)

            # unfortunately pandas dataframe sparse do not allow item assignment :( so we need to work on numpy array
            # first get the numpy array in correct order
            varnames = forest.varnames
            X_ar = X.loc[:, varnames].values

            self._maps = maps = dict()

            for parent, child in forest.get_arcs(names=False):
                # assert (parent, child) not in maps, "Error: edge already exists"

                # levels_mapping_df = X.loc[:, (parent, child)].groupby(parent).agg(lambda x: x.value_counts().index[0])
                # maps[parent, child] = levels_mapping_df[child].to_dict()
                pc_df = pd.DataFrame(X_ar[:, (parent, child)], columns=["parent", "child"])
                levels_mapping_df = pc_df.groupby("parent").agg(_get_most_common_value)

                # Init the dict for parent if it does not exit
                maps.setdefault(parent, dict())

                # Fill the parent-child item with the mapping object
                maps[parent][child] = ParentChildMapping(levels_mapping_df.iloc[:, 0].to_dict())

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

        self.assert_valid_input(X, df_extras_allowed=True)

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
                X[:, child] = self._maps[parent][child].predict_child_from_parent_ar(X[:, parent])
        else:
            if not inplace:
                X = X.copy()

            # walk the tree from the roots
            varnames = forest.varnames
            for _, parent, child in forest.walk_arcs(names=False):
                X.loc[:, varnames[child]] = self._maps[parent][child].predict_child_from_parent(X.loc[:, varnames[parent]])

        if not inplace:
            return X
