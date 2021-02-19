# -*- coding: utf-8 -*-
# The above encoding declaration is needed because we use non-scii characters in `get_trees_str_list`.
# The below import transforms automatically all strings in this file in unicode in python 2.
from __future__ import unicode_literals  # See tests/encoding_ref_help.py for a detailed explanation
import numpy as np
import pandas as pd

from pyitlib import discrete_random_variable as drv

from qdscreen.compat import encode_if_py2

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
    """A quasi-deterministic forest returned by `qd_screen`"""
    __slots__ = ('_adjmat',         # a square numpy array or pandas DataFrame containing the adjacency matrix (parent->child)
                 '_parents',        # a 1d np array or a pandas Series relating each child to its parent index or -1 if a root
                 'is_nparray',      # a boolean indicating if this was built from numpy array (and not pandas dataframe)
                 '_roots_mask',     # a 1d np array or pd Series containing a boolean mask for root variables
                 '_roots_wc_mask',  # a 1d np array or pd Series containing a boolean mask for root with children
                 'stats'            # an optional `Entropies` object stored for debug
                 )

    def __init__(self,
                 adjmat=None,    # type: Union[np.ndarray, pd.DataFrame]
                 parents=None,   # type: Union[np.ndarray, pd.Series]
                 stats=None      # type: Entropies
                 ):
        """

        :param adjmat:
        :param parents:
        """
        self.stats = stats
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
        self._roots_wc_mask = None

    def assert_stats_kept(self):
        if self.stats is None:
            raise ValueError("`stats` are not available in this QDForest instance. If you created it using `qd_screen`,"
                             " you should use `keep_stats=True`.")

    def assert_pandas_capable(self):
        """Utility method to assert that "not self.is_nparray" """
        if self.is_nparray:
            raise ValueError("This QDForest instance was built with numpy arrays, it is not pandas compliant")

    def assert_names_available(self):
        """"""
        if self.is_nparray:
            raise ValueError("Variable names are not available")

    def _validate_names_arg(self,
                            names):
        """Utility method to validate and set default value of the `names` argument used in most methods"""
        if names is None:
            names = not self.is_nparray
        if names:
            self.assert_names_available()
        return names

    @property
    def nb_vars(self):
        return self._parents.shape[0] if self._parents is not None else self._adjmat.shape[0]

    @property
    def varnames(self):
        self.assert_names_available()
        return np.array(self._adjmat.columns) if self._adjmat is not None else np.array(self._parents.index)

    def indices_to_mask(self,
                        indices,
                        in_names=None,
                        out_names=None,
                        ):
        """Utility to convert a list of indices to a numpy or pandas mask"""
        in_names = self._validate_names_arg(in_names)
        out_names = self._validate_names_arg(out_names)
        mask = np.zeros((self.nb_vars,), dtype=bool)
        if out_names:
            mask = pd.Series(mask, index=self.varnames, dtype=bool)
        if in_names and not out_names:
            # TODO
            mask[self.varnames] = True
        elif not in_names and out_names:
            mask.iloc[indices] = True
        else:
            mask[indices] = True
        return mask

    def mask_to_indices(self, mask):
        """Utili"""
        if isinstance(mask, np.ndarray):
            return np.where(mask)
        else:
            return mask[mask].index

    @property
    def adjmat_ar(self):
        """The adjacency matrix as a 2D numpy array"""
        return self.adjmat if self.is_nparray else  self.adjmat.values

    @property
    def adjmat(self):
        """The adjacency matrix as a pandas DataFrame or a 2D numpy array"""
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
        """A numpy array containing the indices of all parent nodes"""
        return self.parents if self.is_nparray else self.parents['idx'].values

    @property
    def parents(self):
        """A numpy array containing the indices of all parent nodes, or a pandas DataFrame containing 'idx' and 'name'"""
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
    def roots_mask_ar(self):
        """A 1D numpy mask array indicating if a node is a root node"""
        return self.roots_mask if self.is_nparray else self.roots_mask.values

    @property
    def roots_mask(self):
        """A pandas Series or a 1D numpy mask array indicating if a node is a root node"""
        if self._roots_mask is None:
            if self._adjmat is not None:
                self._roots_mask = ~self.adjmat.any(axis=0)
            else:
                self._roots_mask = (self.parents if self.is_nparray else self.parents['idx']) < 0
                if not self.is_nparray:
                    # remove the name of the series
                    self._roots_mask.name = None
        return self._roots_mask

    @property
    def roots_ar(self):
        """A 1D numpy array of root indices"""
        return np.where(self.roots_mask_ar)[0]

    @property
    def roots(self):
        """A pandas Series of root names or 1D numpy array of root indices"""
        if self.is_nparray:
            return self.roots_ar
        else:
            return self.roots_mask.index[self.roots_mask]

    def get_roots(self,
                  names=None  # type: bool
                  ):
        """
        Returns the list of root indices or root names

        :param names: an optional boolean indicating if this method should return names instead of indices. By
            default `names` is set to `not self.is_np`
        :return:
        """
        names = self._validate_names_arg(names)
        return list(self.roots) if names else list(self.roots_ar)

    @property
    def non_roots_ar(self):
        """A 1D numpy array of non-root indices"""
        return np.where(~self.roots_mask_ar)[0]

    @property
    def non_roots(self):
        """A pandas Series of non-root names or 1D numpy array of non-root indices"""
        if self.is_nparray:
            return self.non_roots_ar
        else:
            return self.roots_mask.index[~self.roots_mask]

    def get_non_roots(self,
                      names=None  # type: bool
                      ):
        """
        Returns the list of non-root indices or names

        :param names: an optional boolean indicating if this method should return names instead of indices. By
            default `names` is set to `not self.is_np`
        :return:
        """
        names = self._validate_names_arg(names)
        return list(self.non_roots) if names else list(self.non_roots_ar)

    @property
    def nodes_with_children_mask_ar(self):
        """A 1D numpy mask array indicating nodes that have at least 1 child"""
        return self.nodes_with_children_mask if self.is_nparray else self.nodes_with_children_mask.values

    @property
    def nodes_with_children_mask(self):
        """A pandas Series or 1D numpy array indicating nodes that have at least 1 child"""
        if self._adjmat is not None:
            return self._adjmat.any(axis=1)
        else:
            nwc_idx = np.unique(self.parents_indices_ar[self.parents_indices_ar >= 0])
            return self.indices_to_mask(nwc_idx)

    @property
    def roots_with_children_mask_ar(self):
        """A 1D numpy mask array indicating root nodes that have at least 1 child"""
        return self.roots_with_children_mask if self.is_nparray else self.roots_with_children_mask.values

    @property
    def roots_with_children_mask(self):
        """A pandas Series or 1D numpy mask array indicating root nodes that have at least 1 child"""
        if self._roots_wc_mask is None:
            if self._adjmat is not None:
                # a root with children = a root AND a node with children
                self._roots_wc_mask = self._roots_mask & self.nodes_with_children_mask
            else:
                # a root with children = a parent of a level 1 child node
                level_1_nodes_mask = ((self.parents_indices_ar >= 0)
                                      & (self.parents_indices_ar[self.parents_indices_ar] == -1))
                rwc_indices = np.unique(self.parents_indices_ar[level_1_nodes_mask])
                self._roots_wc_mask = self.indices_to_mask(rwc_indices, in_names=False, out_names=not self.is_nparray)
        return self._roots_wc_mask

    @property
    def roots_with_children_ar(self):
        """A 1D numpy array with the indices of root nodes that have at least 1 child"""
        return np.where(self.roots_with_children_mask_ar)[0]

    @property
    def roots_with_children(self):
        """A pandas Series or 1D numpy array with names/indices of root nodes that have at least 1 child"""
        if self.is_nparray:
            return self.roots_with_children_ar
        else:
            return self.roots_with_children_mask.index[self.roots_with_children_mask]

    def get_roots_with_children(self,
                                names=None  # type: bool
                                ):
        """
        Returns the list of root with children

        :param names: an optional boolean indicating if this method should return names instead of indices. By
            default `names` is set to `not self.is_np`
        :return:
        """
        names = self._validate_names_arg(names)
        return list(self.roots_with_children) if names else list(self.roots_with_children_ar)

    def get_children(self,
                     parent_idx=None,   # type: int
                     parent_name=None,  # type: str
                     names=None         # type: bool
                     ):
        """
        Returns the list of children of a node

        :param parent_idx: the index of the node to query for
        :param parent_name: the name of the node to query for
        :param names: a boolean to indicate if the returned children should be identified by their names (`True`) or
            indices (`False`). By default if the parent is identified with `parent_idx` indices will be returned (`False`),
            and if the parent is identified with parent_name names will be returned (`True`)
        :return: an array containing indices of the child nodes
        """

        if parent_idx is not None:
            if parent_name is not None:
                raise ValueError("Only one of `parent_idx` and `parent_name` should be provided")
            node = parent_idx
            if names is None:
                names = False
        elif parent_name is not None:
            node = parent_name
            if names is None:
                names = True
            self.assert_names_available()
        else:
            raise ValueError("You should provide a non-None `parent_idx` or `parent_name`")

        if self.is_nparray:
            return np.where(self.parents == node)[0]
        else:
            qcol = 'idx' if parent_idx is not None else 'name'
            if not names:
                return np.where(self.parents[qcol] == node)[0]
            else:
                return self.parents[self.parents[qcol] == node].index.values

    def get_arcs(self,
                 multiindex=False,  # type: bool
                 names=None         # type: bool
                 ):
        # type: (...) -> Union[Iterable[Tuple[int, int]], Iterable[Tuple[str, str]], Tuple[Iterable[int], Iterable[int]], Tuple[Iterable[str], Iterable[str]]]
        """
        Return the arcs of an adjacency matrix, an iterable of (parent, child) indices or names

        If 'multiindex' is True instead of returning an iterable of (parent, child), it returns a tuple of iterables
        (all the parents, all the childs).

        :param A:
        :param multiindex: if this is True, a 2-tuple of iterable is returned instead of an iterable of 2-tuples
        :param names: an optional boolean indicating if this method should return names instead of indices. By
            default `names` is set to `not self.is_np`
        :return:
        """
        names = self._validate_names_arg(names)
        if self._adjmat is not None:
            return get_arcs_from_adjmat(self._adjmat, multiindex=multiindex, names=names)
        else:
            return get_arcs_from_parents_indices(self._parents, multiindex=multiindex, names=names)

    def walk_arcs(self,
                  parent_idx=None,     # type: int
                  parent_name=None,    # type: str
                  names=None           # type: bool
                  ):
        """
        Yields a sequence of (parent, child) indices or names. As opposed to `get_arcs` the sequence follows the tree
        order: starting from the list of root nodes, for every node, it is returned first and then all of its children.
        (depth first, not breadth first)

        :param parent_idx: the optional index of a node. If provided, only the subtree behind this node will be walked
        :param parent_name: the optional name of a node. If provided, only the subtree behind this node will be walked
        :param names: an optional boolean indicating if this method should return names instead of indices. By
            default `names` is set to `not self.is_np`
        :return: yields a sequence of (level, i, j)
        """
        names = self._validate_names_arg(names)
        if names:
            get_children = lambda node: self.get_children(parent_name=node)
        else:
            get_children = lambda node: self.get_children(parent_idx=node)

        def _walk(from_node, level):
            for child in get_children(from_node):
                yield level, from_node, child
                for l, i, j in _walk(child, level+1):
                    yield l, i, j

        if parent_idx is not None:
            if parent_name is not None:
                raise ValueError("Only one of `parent_idx` and `parent_name` should be provided")
            root_nodes = (parent_idx,) if not names else (self.parents.index[parent_idx],)
        elif parent_name is not None:
            root_nodes = (parent_name,)
        else:
            root_nodes = (self.roots if names else self.roots_ar)

        for n in root_nodes:
            # walk the subtree
            for level, i, j in _walk(n, level=0):
                yield level, i, j

    # ------- display methods

    def to_str(self,
               names=None,  # type: bool
               mode="headers"  # type: str
               ):
        """
        Provides a string representation of this forest.

        :param names: an optional boolean indicating if this method should return names instead of indices. By
            default `names` is set to `not self.is_np`
        :param mode: one of "compact", "headers", and "full" (displays the trees)
        :return:
        """
        names = self._validate_names_arg(names)
        nb_vars = self.nb_vars
        roots = self.get_roots(names)
        non_roots = self.get_non_roots(names)
        nb_roots = len(roots)
        nb_arcs = nb_vars - nb_roots

        roots_with_children = self.get_roots_with_children(names)
        nb_roots_with_children = len(roots_with_children)

        nb_sole_roots = (nb_roots - nb_roots_with_children)

        if mode == "compact":
            return "QDForest (%s vars = %s roots + %s determined by %s of the roots)" \
                   % (nb_vars, nb_roots, nb_arcs, nb_roots_with_children)

        # list of root node indice/name with a star when they have children
        roots_str = [("%s*" if r in roots_with_children else "%s") % r for r in roots]
        # list of
        non_roots_str = ["%s" % r for r in non_roots]

        headers = "\n".join((
                "QDForest (%s vars):" % nb_vars,
                " - %s roots (%s+%s*): %s" % (nb_roots, nb_sole_roots, nb_roots_with_children, ", ".join(roots_str)),
                " - %s other nodes: %s" % (nb_arcs, ", ".join(non_roots_str)),
        ))
        if mode == "headers":
            return headers
        elif mode == "full":
            tree_str = "\n".join(self.get_trees_str_list())
            return "%s\n\n%s" % (headers, tree_str)
        else:
            raise ValueError("Unknown mode: %r" % mode)

    @encode_if_py2
    def __str__(self):
        """ String representation, listing all useful information when available """
        if self.nb_vars > 30:
            return self.to_str(mode="headers")
        else:
            return self.to_str(mode="full")

    @encode_if_py2
    def __repr__(self):
        # return str(self)  # note if we use this then we'll have to comment the decorator
        return self.to_str(mode="compact")

    def print_arcs(self,
                   names=None  # type: bool
                   ):
        """ Prints the various deterministic arcs in this forest """
        print("\n".join(self.get_arcs_str_list(names=names)))

    def get_arcs_str_list(self,
                          names=None  # type: bool
                          ):
        """ Returns a list of string representation of the various deterministic arcs in this forest """
        res_str_list = []
        for parent, child in self.get_arcs(names=names):
            res_str_list.append("%s -> %s" % (parent, child))
        return res_str_list

    def get_trees_str_list(self,
                           all=False,
                           names=None  # type: bool
                           ):
        """
        Returns a list of string representation of the various trees in this forest
        TODO maybe use https://github.com/caesar0301/treelib ? Or networkX ?

        :param all: if True, this will return also the trees that are consistuted of one node
        :param names:
        :return:
        """
        names = self._validate_names_arg(names)
        res_str_list = []
        if all:
            roots = self.get_roots(names)
        else:
            roots = self.get_roots_with_children(names)

        if names:
            walk_arcs = lambda n: self.walk_arcs(parent_name=n)
        else:
            walk_arcs = lambda n: self.walk_arcs(parent_idx=n)

        for r in roots:
            subtree_str = "%s" % r
            for level, _, j in walk_arcs(r):
                subtree_str += "\n%s└─ %s" % ("   " * level, j)
            res_str_list.append(subtree_str + "\n")
        return res_str_list

    # -----------

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

    # --------- tools for entropies analysis to find interesting thresholds

    def get_entropies_table(self,
                            from_to=True,  # type: bool
                            sort_by="from",  # type: str
                            drop_self_link=True,  # type: bool
                            ):
        """ See `Entropies.get_entropies_table` """

        self.assert_stats_kept()
        return self.stats.get_entropies_table(from_to=from_to, sort_by=sort_by, drop_self_link=drop_self_link)

    def plot_increasing_entropies(self):
        """ See `Entropies.plot_increasing_entropies` """

        self.assert_stats_kept()
        self.stats.plot_increasing_entropies()


def qd_screen(X,                  # type: Union[pd.DataFrame, np.ndarray]
              absolute_eps=None,  # type: float
              relative_eps=None,  # type: float
              keep_stats=False    # type: bool
              ):
    # type: (...) -> QDForest
    """
    Finds the (quasi-)deterministic relationships (functional dependencies) between the variables in `X`, and returns
    a `QDForest` object representing the forest of (quasi-)deterministic trees. This object can then be used to fit a
    feature selection model or to learn a Bayesian Network structure.

    By default only deterministic relationships are detected. Quasi-determinism can be enabled by setting either
    an threshold on conditional entropies (`absolute_eps`) or on relative conditional entropies (`relative_eps`). Only
    one of them should be set.

    By default (`keep_stats=False`) the entropies tables are not preserved once the forest has been created. If you wish
    to keep them available, set `keep_stats=True`. The entropies tables are then available in the `<QDForest>.stats`
    attribute, and threshold analysis methods such as `<QDForest>.get_entropies_table(...)` and
    `<QDForest>.plot_increasing_entropies()` become available.

    :param X: the dataset as a pandas DataFrame or a numpy array. Columns represent the features to compare.
    :param absolute_eps: Absolute entropy threshold. Any feature `Y` that can be predicted from another feature `X` in
        a quasi-deterministic way, that is, where conditional entropy `H(Y|X) <= absolute_eps`, will be removed. The
        default value is `0` and corresponds to removing deterministic relationships only.
    :param relative_eps: Relative entropy threshold. Any feature `Y` that can be predicted from another feature `X` in
        a quasi-deterministic way, that is, where relative conditional entropy `H(Y|X)/H(Y) <= relative_eps` (a value
        between `0` and `1`), will be removed. Only one of `absolute_eps` or `relative_eps` should be provided.
    :param keep_stats: a boolean indicating if the various entropies tables computed in the process should be kept in
        memory in the resulting forest object (`<QDForest>.stats`), for further analysis. By default this is `False`.
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
    A_noredundancy = remove_redundancies(A_orig, selection_order=None if is_strict else X_stats.list_vars_by_entropy_order(desc=True))

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
    qd_forest = QDForest(parents=parents, stats=X_stats if keep_stats else None)

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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        res = """Statistics computed for dataset:
%s
...(%s rows)

Entropies (H):
%s

Conditional entropies (Hcond = H(row|col)):
%s

Relative conditional entropies (Hcond_rel = H(row|col)/H(row)):
%s
""" % (self.dataset.head(2), len(self.dataset), self.H.T, self.Hcond, self.Hcond_rel)
        return res

    @property
    def dataset_df(self):
        """The dataset as a pandas DataFrame, if pandas is available"""
        if self.is_nparray:
            # see https://numpy.org/doc/stable/user/basics.rec.html#manipulating-and-displaying-structured-datatypes
            if self.dataset.dtype.names is not None:
                # structured array
                self._dataset_df = pd.DataFrame(self.dataset)
            else:
                # unstructured array... same ?
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

    entropies_ar = H_ar
    """An alias for H_ar"""

    @property
    def H(self):
        """The entropies of all variables. a pandas Series if df was a pandas dataframe, else a 1D numpy array"""
        if self._H is None:
            # Using pyitlib to compute H (hopefully efficiently)
            # Unfortunately this does not work with numpy arrays, convert to pandas TODO report
            # note: we convert to string type to avoid a bug with ints. TODO...
            self._H = drv.entropy(self.dataset_df.T.astype(str))
            if not self.is_nparray:
                self._H = pd.Series(self._H, index=self.varnames)

            # basic sanity check: should all be positive
            assert np.all(self._H >= 0)
        return self._H

    entropies = H
    """An alias for H"""

    @property
    def Hcond_ar(self):
        """ The conditional entropy matrix (i, j) = H(Xi | Xj) as a numpy array """
        return self.Hcond if self.is_nparray else self.Hcond.values

    conditional_entropies_ar = Hcond_ar
    """An alias for Hcond_ar"""

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
            # note: we convert to string type to avoid a bug with ints. TODO...
            self._Hcond = drv.entropy_conditional(self.dataset_df.T.astype(str))

            # add the row/column headers
            if not self.is_nparray:
                self._Hcond = pd.DataFrame(self._Hcond, index=self.varnames, columns=self.varnames)

            # basic sanity check: should all be positive
            assert np.all(self._Hcond >= 0)
        return self._Hcond

    conditional_entropies = Hcond
    """An alias for Hcond"""

    @property
    def Hcond_rel_ar(self):
        """ The relative conditional entropy matrix (i, j) = H(Xi | Xj) / H(Xi) as a numpy array """
        return self.Hcond_rel if self.is_nparray else self.Hcond_rel.values

    relative_conditional_entropies_ar = Hcond_rel_ar
    """An alias for Hcond_rel_ar"""

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

    relative_conditional_entropies = Hcond_rel
    """An alias for Hcond_rel"""

    def list_vars_by_entropy_order(self, desc=False):
        """
        Returns the indices or names of variables in ascending (resp. descending) order of entropy

        :param desc: if True the order is descending, else ascending
        :return:
        """
        if self.is_nparray:
            return (-self.H).argsort() if desc else self.H.argsort()
        else:
            return (-self.H.values).argsort() if desc else self.H.values.argsort()

    # --------- tools for entropies analysis to find interesting thresholds

    def get_entropies_table(self,
                            from_to=True,    # type: bool
                            sort_by="from",  # type: str
                            drop_self_link=True,  # type: bool
                            ):
        """
        Returns a pandas series or numpy array with four columns: from, to, cond_entropy, rel_cond_entropy.
        The index is 'arc', a string representing the arc e.g. "X->Y".

        :param from_to: a boolean flag indicating if 'from' and 'to' columns should remain in the returned dataframe
            (True, default) or be dropped (False)
        :param sort_by: a string indicating if the arcs should be sorted by origin ("from", default) or destination
            ("to"), or by value "rel_cond_entropy" or "cond_entropy", in the resulting table.
        :param drop_self_link: by default node-to-self arcs are not returned in the list. Turn this to False to include
            them.
        :return:
        """
        if self.is_nparray:
            raise NotImplementedError("TODO")
        else:
            # 1. gather the data and unpivot it so that there is one row per arc
            res_df = pd.DataFrame({
                'cond_entropy': self.Hcond.unstack(),
                'rel_cond_entropy': self.Hcond_rel.unstack(),
            })
            res_df.index.names = ['from', 'to']
            res_df = res_df.reset_index()

            # 2. filter out node-to-self if needed
            if drop_self_link:
                res_df = res_df.loc[res_df['from'] != res_df['to']].copy()

            # 3. create the arc names column and make it the index
            def make_arc_str(row):
                return "%s->%s" % (row['from'], row.to)  # note that .from is a reserved python symbol !
            res_df['arc'] = res_df.apply(make_arc_str, axis=1)
            res_df.set_index('arc', inplace=True)

            # 4. Optionally sort differently
            all_possibilities = ("from", "to", "cond_entropy", "rel_cond_entropy")
            if sort_by == all_possibilities[0]:
                pass  # already done
            elif sort_by in all_possibilities[1:]:
                res_df.sort_values(by=sort_by, inplace=True)
            else:
                raise ValueError("Invalid value for `sort_by`: %r. Possible values: %r" % (sort_by, all_possibilities))

            # 5. Optionally drop the from and to columns
            if not from_to:
                res_df.drop(['from', 'to'], axis=1, inplace=True)

            return res_df

    def plot_increasing_entropies(self):
        """

        :return:
        """
        import matplotlib.pyplot as plt

        # get the dataframe
        df = self.get_entropies_table(sort_by="rel_cond_entropy")

        # plot with all ticks
        df_sorted = df["rel_cond_entropy"]  # .sort_values()
        df_sorted.plot(title="Relative conditional entropy H(X|Y)/H(X) for each arc X->Y, by increasing order",
                       figsize=(15, 10))
        plt.xticks(np.arange(len(df_sorted)), df_sorted.index, rotation=90)


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
    is_np_array = isinstance(parents, np.ndarray)
    if not names:
        if not is_np_array:
            # assume a dataframe with an 'idx' column as in QDForest class
            parents = parents['idx'].values
        n = len(parents)
        childs_mask = parents >= 0
        res = parents[childs_mask], np.arange(n)[childs_mask]
        return res if multiindex else zip(*res)
    else:
        if is_np_array:
            raise ValueError("Names are not available, this is a numpy array")
        else:
            #     cols = A.columns
            #     return ((cols[i], cols[j]) for i, j in zip(*np.where(A)))
            res = parents.loc[parents['name'].notna(), 'name']
            res = res.values, res.index
            return res if multiindex else zip(*res)


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
            raise ValueError("Names are not available, this is a numpy array")
        else:
            cols = A.columns
            res_ar = np.where(A)
            if multiindex:
                return ((cols[i] for i in l) for l in res_ar)  # noqa
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
