from functools import partial

import numpy as np
import pandas as pd

from pyitlib import discrete_random_variable as drv

try:
    from typing import Union, Iterable, Tuple
except:  # noqa
    pass


# TODO
# Cramer's V + Theil's U:
#    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#    https://stackoverflow.com/questions/46498455/categorical-features-correlation
#    https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix (+ corrected version)
#    https://stackoverflow.com/questions/61236715/correlation-between-categorical-variables-within-a-dataset
#
# https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables-pandas
# https://stackoverflow.com/questions/44694228/how-to-check-for-correlation-among-continuous-and-categorical-variables-in-pytho
#

def _add_names_to_parents_idx_series(parents):
    parents = pd.DataFrame(parents, columns=('idx',))
    parents['name'] = parents.index[parents['idx']].where(parents['idx'] >= 0, None)
    return parents


class QDForest(object):
    """A quasi-deterministic forest returned by `qdeterscreen`"""
    __slots__ = ('_adjmat',   # a square numpy array or pandas DataFrame containing the adjacency matrix (parent->child)
                 '_parents',  # a 1d np array or a pandas Series relating each child to its parent index or -1 if a root
                 'is_nparray',  # a boolean indicating if this was built from numpy array (and not pandas dataframe)
                 '_roots_mask', # a 1d np array or pd Series containing a boolean mask for root variables
                 )

    def __init__(self,
                 adjmat=None,   # type: Union[np.ndarray, pd.DataFrame]
                 parents=None   # type: Union[np.ndarray, pd.Series]
                 ):
        """

        :param adjmat:
        :param parents:
        """
        self._adjmat = adjmat
        self.is_nparray = isinstance((adjmat if adjmat is not None else parents), np.ndarray)
        if parents is not None:
            if not self.is_nparray:
                # create a df with two columns: indices and names
                parents = _add_names_to_parents_idx_series(parents)
            self._parents = parents
        else:
            self._parents = None

        self._roots_mask = None

    @property
    def nb_vars(self):
        return self._parents.shape[0] if self._parents is not None else self._adjmat.shape[0]

    @property
    def varnames(self):
        if self.is_nparray:
            raise ValueError("Variable names not available")
        return np.array(self._adjmat.columns) if self._adjmat is not None else np.array(self._parents.index)

    def get_children(self, node, is_node_idx=None, return_names=None):
        """

        :param node: if n
        :return: an array containing indices of the child nodes
        """
        if is_node_idx is None:
            is_node_idx = self.is_nparray
        if return_names is None:
            return_names = not self.is_nparray

        if self.is_nparray:
            return np.where(self.parents == node)[0]
        else:
            qcol = 'idx' if is_node_idx else 'name'
            if not return_names:
                return np.where(self.parents[qcol] == node)[0]
            else:
                return self.parents[self.parents[qcol] == node].index.values

    def get_arcs(self, multiindex=False, names=None):
        """

        :param names:
        :return:
        """
        if names is None:
            names = not self.is_nparray
        if self._adjmat is not None:
            return get_arcs_from_adjmat(self._adjmat, multiindex=multiindex, names=names)
        else:
            return get_arcs_from_parents_indices(self._parents, multiindex=multiindex, names=names)

    @property
    def adjmat_ar(self):
        return self.adjmat if self.is_nparray else  self.adjmat.values

    @property
    def adjmat(self):
        if self._adjmat is None:
            # compute adjmat from parents.
            n = self.nb_vars
            adjmat = np.zeros((n, n), dtype=bool)
            # from https://stackoverflow.com/a/46018613/7262247
            index_array = get_arcs_from_parents_indices(self.parents_indices_ar, multiindex=True)
            flat_index_array = np.ravel_multi_index(index_array, adjmat.shape)
            np.ravel(adjmat)[flat_index_array] = True
            if self.is_nparray:
                self._adjmat = adjmat
            else:
                self._adjmat = pd.DataFrame(adjmat, index=self.varnames, columns=self.varnames)
        return self._adjmat

    @property
    def parents_indices_ar(self):
        return self.parents if self.is_nparray else self.parents['idx'].values

    @property
    def parents(self):
        if self._parents is None:
            # compute parents from adjmat, whether a dataframe or an array
            # TODO maybe use a sparse array here?
            parents = to_forest_parents_indices(self._adjmat)
            if not self.is_nparray:
                # create a df with two columns: indices and names
                parents = _add_names_to_parents_idx_series(parents)
            self._parents = parents
        return self._parents

    @property
    def roots_ar(self):
        return np.where(self.roots_mask_ar)[0]

    @property
    def roots(self):
        """"""
        if self.is_nparray:
            return self.roots_ar
        else:
            return self.roots_mask.index[self.roots_mask]

    @property
    def roots_mask_ar(self):
        return self.roots_mask if self.is_nparray else self.roots_mask.values

    @property
    def roots_mask(self):
        """Lazily computed roots"""
        if self._roots_mask is None:
            if self._adjmat is not None:
                self._roots_mask = ~self.adjmat.any(axis=0)
            else:
                self._roots_mask = (self.parents if self.is_nparray else self.parents['idx']) < 0
        return self._roots_mask

    def walk_arcs(self, return_names=None):
        """yields a sequence of (parent, child) indices or names"""

        if return_names is None:
            return_names = not self.is_nparray

        get_children = partial(self.get_children, is_node_idx=not return_names, return_names=return_names)

        def _walk(from_node):
            for child in get_children(from_node):
                yield from_node, child
                for i, j in _walk(child):
                    yield i, j

        for n in (self.roots if return_names else self.roots_ar):
            # walk the subtree
            for i, j in _walk(n):
                yield i, j

    def fit_selector_model(self,
                           X  # type: Union[np.ndarray, pd.DataFrame]
                           ):
        # type: (...) -> QDSelectorModel
        """
        Returns a new instance of `QDSelectorModel` fit with the data in `X`

        :param X:
        :return:
        """
        from .selector import QDSelectorModel
        model = QDSelectorModel(self)
        model.fit(X)
        return model


def qdeterscreen(X,                  # type: Union[pd.DataFrame, np.ndarray]
                 absolute_eps=None,  # type: float
                 relative_eps=None,  # type: float
                 ):
    # type: (...) -> QDForest
    """
    Finds the (quasi-)deterministic relationships between the variables in X, and returns the adjacency matrix of the
    resulting forest of (quasi-)deterministic trees.

    :param X: the dataset as a pandas DataFrame or a numpy array. Columns represent the features to compare.
    :param absolute_eps: Absolute entropy threshold. Any feature Y that can be predicted from
        another feature X in a quasi-deterministic way, that is, where conditional entropy H(Y|X) <= absolute_eps,
        will be removed. The default value is 0 and corresponds to removing deterministic relationships only.
    :param relative_eps: Relative entropy threshold. Any feature Y that can be predicted from another feature X in a
        quasi-deterministic way, that is, where relative conditional entropy H(Y|X)/H(Y) <= relative_eps (between 0
        and 1), will be removed. Only one of absolute_eps and relative_eps should be provided.
    :return:
    """
    # only work on the categorical features
    X = get_categorical_features(X)

    # sanity check
    if len(X) == 0:
        raise ValueError("Empty dataset provided")

    # parameters check and defaults
    if absolute_eps is None:
        if relative_eps is None:
            # nothing is provided, use absolute threshold 0
            absolute_eps = 0
            is_absolute = True
        else:
            # relative threshold provided
            is_absolute = False
    else:
        if relative_eps is not None:
            raise ValueError("only one of absolute and relative should be passed")
        # absolute threshold provided
        is_absolute = True

    is_strict = (absolute_eps == 0.) if is_absolute else (relative_eps == 0.)

    # sanity check
    if is_absolute and absolute_eps < 0:
        raise ValueError("epsilon_absolute should be positive")
    elif not is_absolute and (relative_eps < 0 or relative_eps > 1):
        raise ValueError("epsilon_relative should be 0=<eps=<1")

    # (0) compute conditional entropies or relative conditional entropies
    A_orig, X_stats = get_adjacency_matrix(X, eps_absolute=absolute_eps, eps_relative=relative_eps)

    # (2) identify redundancy
    A_noredundancy = remove_redundancies(A_orig, selection_order=None if is_strict else X_stats.entropy_order_desc)

    # (3) transform into forest: remove extra parents by keeping only the parent with lowest entropy / nb levels ?
    # if X -> Z and X -> Y and Y -> Z then Z has two parents X and Y but only Y should be kept
    # Determinist case: minimize the number of parameters: take the minimal nb of levels
    # Quasi-determinist case: take the lowest entropy
    entropy_order_inc = (X_stats.H_ar).argsort()
    # parent_order = compute_nb_levels(df) if is_strict else entropy_based_order
    # A_forest = to_forest_adjmat(A_noredundancy, entropy_order_inc)
    # qd_forest = QDForest(adjmat=A_forest)
    parents = to_forest_parents_indices(A_noredundancy, entropy_order_inc)
    # if not X_stats.is_nparray:
    #     parents = pd.Series(parents, index=A_noredundancy.columns)
    qd_forest = QDForest(parents=parents)

    # forestAdjMatList <- FromHToDeterForestAdjMat(H = H, criterion = criterionNlevels, removePerfectMatchingCol = removePerfectMatchingCol)

    # (4) convert to pgmpy format
    # deterForest <- FromAdjMatToBnLearn(adjMat = forestAdjMat)
    # rootsF <- Roots(forestAdjMat)

    # if(length(rootsFNames) == 1){
    #     print("Only one tree in the forest !!! No root graph computed")
    #     print("****************")
    #     print("Training - Phase 2 and Phase 3 obsolete")
    #     print("****************")
    #     gstar <- deterForest
    #     G_R <- NULL
    #     print("Final graph computed")
    #   }else{
    #     print(paste("Multiple trees in the forest (", length(rootsFNames),") Root graph will be computed.", sep = "", collapse = ""))
    #     rootsOnly <- ReduceDataAndH(variablesToKeep = rootsFNames, dataFrame = data, H)
    #     dataRoots <- rootsOnly[[1]]
    #     HRoots <- rootsOnly[[2]]
    #     rm(rootsOnly)
    #     print("****************")
    #     print("Training - Phase 2: computation of the root graph")
    #     print("****************")
    #     # removePerfectlyMatchingCol useless here because either pmc are in the same trees (but only one is the root), either they were deleted earlier on
    #     G_R <- SotABNsl(data = dataRoots, method = method, score = score, hyperparamList, removePerfectMatchingCol = removePerfectMatchingCol, ...)
    #     edgeListG_R <- FromBnToEdgeList(bn = G_R)
    #     colnames(edgeListG_R) <- c('from', 'to')
    #     print("****************")
    #     print("Training - Phase 3: fusion of the deter forest with the root graph")
    #     print("****************")
    #     gstarAdjMat <- AddEdgesFromEdgeList(originalAdjMatrix = forestAdjMat, edgeListToAdd = edgeListG_R)
    #     gstar <- FromAdjMatToBnLearn(gstarAdjMat)
    #     print("Final graph computed")
    #   }

    return qd_forest


class Entropies(object):
    """ to do this could be easier to read with pyfields and default value factories """

    __slots__ = ('dataset', 'is_nparray', '_H', '_Hcond', '_Hcond_rel', '_dataset_df')

    def __init__(self, df_or_array):
        """

        :param df_or_array: a pandas dataframe or numpy array where the variables are the columns
        """
        self.dataset = df_or_array
        self.is_nparray = isinstance(df_or_array, np.ndarray)
        self._H = None           # type: Union[np.ndarray, pd.Series]
        self._Hcond = None       # type: Union[np.ndarray, pd.DataFrame]
        self._Hcond_rel = None   # type: Union[np.ndarray, pd.DataFrame]
        self._dataset_df = None  # type: pd.DataFrame

    @property
    def dataset_df(self):
        """The dataset as a pandas DataFrame, if pandas is available"""
        if self.is_nparray:
            # see https://numpy.org/doc/stable/user/basics.rec.html#manipulating-and-displaying-structured-datatypes
            if self.dataset.dtype.names is not None:
                # structured array
                self._dataset_df = pd.DataFrame(self.dataset)
            else:
                # unstructured array
                self._dataset_df = pd.DataFrame(self.dataset)
            return self._dataset_df
        else:
            return self.dataset

    @property
    def nb_vars(self):
        return self.dataset.shape[1]

    @property
    def varnames(self):
        if self.is_nparray:
            raise ValueError("Variable names not available")
        return list(self.dataset.columns)

    @property
    def H_ar(self):
        """ The entropy matrix (i, j) = H(Xi | Xj) as a numpy array """
        return self.H if self.is_nparray else self.H.values

    @property
    def H(self):
        """The entropies of all variables. a pandas Series if df was a pandas dataframe, else a 1D numpy array"""
        if self._H is None:
            # Using pyitlib to compute H (hopefully efficiently)
            # Unfortunately this does not work with numpy arrays, convert to pandas TODO report
            self._H = drv.entropy(self.dataset_df.T)
            if not self.is_nparray:
                self._H = pd.Series(self._H, index=self.varnames)

            # basic sanity check: should all be positive
            assert np.all(self._H >= 0)
        return self._H

    @property
    def Hcond_ar(self):
        """ The conditional entropy matrix (i, j) = H(Xi | Xj) as a numpy array """
        return self.Hcond if self.is_nparray else self.Hcond.values

    @property
    def Hcond(self):
        """
        The conditional entropy matrix (i, j) = H(Xi | Xj).
        A pandas Dataframe or a 2D numpy array depending on dataset type
        """
        if self._Hcond is None:
            # Old attempt to do it ourselves
            # (0) init H
            # nb_vars = len(df.columns)
            # H = np.empty((nb_vars, nb_vars), dtype=float)
            # (1) for each column compute the counts per value
            # (2) for each (i, j) pair compute the counts

            # Using pyitlib to compute H (hopefully efficiently)
            # Unfortunately this does not work with numpy arrays, convert to pandas TODO report
            self._Hcond = drv.entropy_conditional(self.dataset_df.T)

            # add the row/column headers
            if not self.is_nparray:
                self._Hcond = pd.DataFrame(self._Hcond, index=self.varnames, columns=self.varnames)

            # basic sanity check: should all be positive
            assert np.all(self._Hcond >= 0)
        return self._Hcond

    @property
    def Hcond_rel_ar(self):
        """ The relative conditional entropy matrix (i, j) = H(Xi | Xj) / H(Xi) as a numpy array """
        return self.Hcond_rel if self.is_nparray else self.Hcond_rel.values

    @property
    def Hcond_rel(self):
        """
        The relative conditional entropy matrix (i, j) = H(Xi | Xj) / H(Xi).
        """
        if self._Hcond_rel is None:
            # compute relative entropy: in each cell (X, Y) we want H(X|Y)/H(X)
            if self.is_nparray:
                self._Hcond_rel = (self.Hcond.T / self.H).T
            else:
                Hrel_array = (self.Hcond.values.T / self.H.values).T
                self._Hcond_rel = pd.DataFrame(Hrel_array, index=list(self.Hcond.columns), columns=list(self.Hcond.columns))

            # basic sanity check
            assert np.all(self._Hcond_rel >= 0.)
            assert np.all(self._Hcond_rel <= 1.)
        return self._Hcond_rel

    @property
    def entropy_order_desc(self):
        if self.is_nparray:
            return (-self.H).argsort()
        else:
            return (-self.H.values).argsort()

    @property
    def entropy_order_asc(self):
        if self.is_nparray:
            return self.H.argsort()
        else:
            return self.H.values.argsort()


def get_adjacency_matrix(df,                 # type: Union[np.ndarray, pd.DataFrame]
                         eps_absolute=None,  # type: float
                         eps_relative=None   # type: float
                         ):
    """

    :param df:
    :param eps_absolute:
    :param eps_relative:
    :return:
    """
    df_stats = Entropies(df)

    # (1) create initial adjacency matrix by thresholding either H(X|Y) (absolute) or H(X|Y)/H(X) (relative)
    if eps_relative is None:
        # default value for eps absolute
        if eps_absolute is None:
            eps_absolute = 0

        # threshold is on H(X|Y)
        edges = (df_stats.Hcond_ar <= eps_absolute).T
    else:
        if eps_absolute is not None:
            raise ValueError("Only one of `eps_absolute` and `eps_relative` should be provided")

        # threshold on H(X|Y)/H(X)
        edges = (df_stats.Hcond_rel_ar <= eps_relative).T

    # the diagonal should be false
    np.fill_diagonal(edges, False)

    # finally create the matrix
    if df_stats.is_nparray:
        A = edges
    else:
        A = pd.DataFrame(edges, index=df_stats.varnames, columns=df_stats.varnames)

    return A, df_stats


def remove_redundancies(A,                    # type: Union[np.ndarray, pd.DataFrame]
                        selection_order=None  # type: np.ndarray
                        ):
    """Cleans the arcs in A between redundant variables.

    A should be a dataframe or numpy array where index = columns and indicate the node name.
    It should contain boolean values where True at (i,j) means there is a directed arc i -> j

    When there are redundant variables (arcs between i->j and j->i), only a single node is kept as the parent
    of all other redundant nodes.

    :param A: an adjacency matrix where A(i, j) indicates that there is an arc i -> j
    :param selection_order: if none (default) the first node in the list will be kept as representant for each
        redundancy class. Otherwise an alternate order (array of indices) can be provided.
    :return:
    """
    assert_adjacency_matrix(A)

    # work on a copy
    is_nparray = isinstance(A, np.ndarray)
    if is_nparray:
        A = A.copy()
        A_df = None
    else:
        A_df = A.copy(deep=True)
        A = A_df.values

    # init
    n_vars = A.shape[0]
    if selection_order is None:
        selection_order = range(n_vars)

    # I contains the list of variable indices to go through. We can remove some in the loop
    I = np.ones((n_vars, ), dtype=bool)

    # for each node
    for i in selection_order:
        # if we did not yet take it into account
        if I[i]:
            # find redundant nodes with this node (= there is an arc in both directions)
            mask_redundant_with_i = A[i, :] * A[:, i]

            # we will stop caring about this redundancy class
            I[mask_redundant_with_i] = False

            # we do not care about i-to-i
            mask_redundant_with_i[i] = False

            # if there is at least one redundant node
            if any(mask_redundant_with_i):
                redundant_nodes_idx = np.where(mask_redundant_with_i)

                # let only the first variable in the list (i) be the parent
                # i becomes the representant of the redundancy class
                A[redundant_nodes_idx, i] = False

                # remove all arcs inter-redundant variables.
                # Note that this destroys the diagonal but we will fix this at the end
                A[redundant_nodes_idx, redundant_nodes_idx] = False

    # restore the diagonal: no
    # np.fill_diagonal(A, True)
    if is_nparray:
        return A
    else:
        return A_df


def to_forest_parents_indices(A,                    # type: Union[np.ndarray, pd.DataFrame]
                              selection_order=None  # type: np.ndarray
                              ):
    # type: (...) -> Union[np.ndarray, pd.DataFrame]
    """ Removes extra arcs in the adjacency matrix A by only keeping the first parent in the given order

    Returns a 1D array of parent index or -1 if root

    :param A: an adjacency matrix, as a dataframe or numpy array
    :param selection_order: an optional order for parent selection. If not provided, the first in the list of columns
        of A will be used
    :return:
    """
    assert_adjacency_matrix(A)
    is_np_array = isinstance(A, np.ndarray)

    # From https://stackoverflow.com/a/47269413/7262247
    if is_np_array:
        mask = A[selection_order, :] if selection_order is not None else A
    else:
        mask = A.iloc[selection_order, :].values if selection_order is not None else A.values

    # return a list containing for each feature, the index of its parent or -1 if it is a root
    indices = np.where(mask.any(axis=0),
                       selection_order[mask.argmax(axis=0)] if selection_order is not None else mask.argmax(axis=0),
                       -1)
    if not is_np_array:
        indices = pd.Series(indices, index=A.columns)
    return indices


def to_forest_adjmat(A,                # type: Union[np.ndarray, pd.DataFrame]
                     selection_order,  # type: np.ndarray
                     inplace=False     # type: bool
                     ):
    # type: (...) -> Union[np.ndarray, pd.DataFrame]
    """ Removes extra arcs in the adjacency matrix A by only keeping the first parent in the given order

    :param A: an adjacency matrix, as a dataframe or numpy array
    :return:
    """
    is_ndarray = isinstance(A, np.ndarray)
    assert_adjacency_matrix(A)

    if not inplace:
        A = A.copy()

    # nb_parents = A_df.sum(axis=0)
    # nodes_with_many_parents = np.where(nb_parents.values > 1)[0]

    # Probably the fastest: when the cumulative nb of parents is above 1 they need to be removed
    if is_ndarray:
        mask_parents_over_the_first = A[selection_order, :].cumsum(axis=0) > 1
        mask_parents_over_the_first = mask_parents_over_the_first[selection_order.argsort(), :]
        A[mask_parents_over_the_first] = False
    else:
        # From https://stackoverflow.com/a/47269413/7262247
        mask_parents_over_the_first = A.iloc[selection_order, :].cumsum(axis=0) > 1
        A[mask_parents_over_the_first] = False

    # # Probably slower but easier to understand
    # # for each column we want to only keep one (or zero) row (parent) with True
    # for i in range(A_df.shape[1]):
    #     first_parent_idx = A_df.values[selection_order, i].argmax(axis=0)
    #     A_df.values[selection_order[first_parent_idx+1:], i] = False

    if not inplace:
        return A


def get_arcs_from_parents_indices(parents,           # type: Union[np.ndarray, pd.DataFrame]
                                  multiindex=False,  # type: bool
                                  names=False        # type: bool
                                  ):
    # type: (...) -> Union[Iterable[Tuple[int, int]], Iterable[Tuple[str, str]], Tuple[Iterable[int], Iterable[int]], Tuple[Iterable[str], Iterable[str]]]
    """
    if multiindex = False ; returns a sequence of pairs : (9, 1), (3, 5), (9, 7)
    if multiindex = True ; returns two sequences of indices: (9, 3, 9), (1, 5, 7)

    :param parents:
    :param multiindex:
    :param names:
    :return:
    """
    if not names:
        n = len(parents)
        childs_mask = parents >= 0
        res = parents[childs_mask], np.arange(n)[childs_mask]
        return res if multiindex else zip(*res)
    else:
        # is_np_array = isinstance(A, np.ndarray)
        # if is_np_array:
        #     raise NotImplementedError()
        # else:
        #     cols = A.columns
        #     return ((cols[i], cols[j]) for i, j in zip(*np.where(A)))
        raise NotImplementedError()


def get_arcs_from_adjmat(A,                 # type: Union[np.ndarray, pd.DataFrame]
                         multiindex=False,  # type: bool
                         names=False        # type: bool
                         ):
    # type: (...) -> Union[Iterable[Tuple[int, int]], Iterable[Tuple[str, str]], Tuple[Iterable[int], Iterable[int]], Tuple[Iterable[str], Iterable[str]]]
    """
    Return the arcs of an adjacency matrix, an iterable of (parent, child) indices or names

    If 'multiindex' is True instead of returning an iterable of (parent, child), it returns a tuple of iterables
    (all the parents, all the childs).

    :param A:
    :param multiindex: if this is True, a 2-tuple of iterable is returned instead of an iterable of 2-tuples
    :param names: if False, indices are returned. Otherwise feature names are returned if any
    :return:
    """
    if not names:
        res = np.where(A)
        return res if multiindex else zip(*res)
    else:
        is_np_array = isinstance(A, np.ndarray)
        if is_np_array:
            raise NotImplementedError()
        else:
            cols = A.columns
            res_ar = np.where(A)
            if multiindex:
                return ((cols[i] for i in l) for l in res_ar)
            else:
                return ((cols[i], cols[j]) for i, j in zip(*res_ar))


def get_categorical_features(df_or_array  # type: Union[np.ndarray, pd.DataFrame]
                             ):
    # type: (...) -> Union[np.ndarray, pd.DataFrame]
    """

    :param df_or_array:
    :return: a dataframe or array with the categorical features
    """
    if isinstance(df_or_array, pd.DataFrame):
        is_categorical_dtype = df_or_array.dtypes.astype(str).isin(["object", "categorical"])
        if not is_categorical_dtype.any():
            raise TypeError("Provided dataframe columns do not contain any categorical datatype (dtype in 'object' or "
                            "'categorical'): found dtypes %r" % df_or_array.dtypes[~is_categorical_dtype].to_dict())
        return df_or_array.loc[:, is_categorical_dtype]
    elif isinstance(df_or_array, np.ndarray):
        # see https://numpy.org/doc/stable/user/basics.rec.html#manipulating-and-displaying-structured-datatypes
        if df_or_array.dtype.names is not None:
            # structured array
            is_categorical_dtype = np.array([str(df_or_array.dtype.fields[n][0]) == "object"
                                             for n in df_or_array.dtype.names])
            if not is_categorical_dtype.any():
                raise TypeError(
                    "Provided dataframe columns do not contain any categorical datatype (dtype in 'object' or "
                    "'categorical'): found dtypes %r" % df_or_array.dtype.fields)
            categorical_names = np.array(df_or_array.dtype.names)[is_categorical_dtype]
            return df_or_array[categorical_names]
        else:
            # non-structured array
            if df_or_array.dtype != np.dtype('O'):
                raise TypeError("Provided array columns are not object nor categorical: found dtype %r"
                                % df_or_array.dtype)
            return df_or_array
    else:
        raise TypeError("Provided data is neither a pd.DataFrame nor a np.ndarray")


def assert_adjacency_matrix(A  # type: Union[np.ndarray, pd.DataFrame]
                            ):
    """Routine to check that A is a proper adjacency matrix"""

    if len(A.shape) != 2:
        raise ValueError("A is not a 2D adjacency matrix, its shape is %sD" % len(A.shape))

    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not a 2D adjacency matrix: it is not square: %r" % A.shape)
