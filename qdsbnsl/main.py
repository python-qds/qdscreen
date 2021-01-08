import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv

try:
    from typing import Union
except:  # noqa
    pass


def qdeterscreen(df,                     # type: pd.DataFrame
                 epsilon_absolute=None,  # type: float
                 epsilon_relative=None,  # type: float
                 ):
    """

    :param df:
    :return:
    """
    # TODO ensure categorical variables or convert

    if epsilon_absolute is None:
        if epsilon_relative is None:
            # if nothing is provided, use absolute threshold 0
            epsilon_absolute = 0
            is_absolute = True
        else:
            # relative threshold provided
            is_absolute = False
    else:
        if epsilon_relative is not None:
            raise ValueError("only one of absolute and relative should be passed")
        # absolute threshold provided
        is_absolute = True

    is_strict = (epsilon_absolute == 0.) if is_absolute else (epsilon_relative == 0.)

    # sanity check
    if is_absolute and epsilon_absolute < 0:
        raise ValueError("epsilon_absolute should be positive")
    elif not is_absolute and (epsilon_relative < 0 or epsilon_relative > 1):
        raise ValueError("epsilon_relative should be 0=<eps=<1")

    # (0) compute conditional entropies or relative conditional entropies
    A_df_orig, df_stats = get_adjacency_matrix(df, eps_absolute=epsilon_absolute, eps_relative=epsilon_relative)

    # (2) identify redundancy
    A_df_noredundancy = remove_redundancies(A_df_orig, selection_order=None if is_strict else df_stats.entropy_order_desc)

    # (3) transform into forest: remove extra parents by keeping only the parent with lowest entropy / nb levels ?
    # if X -> Z and X -> Y and Y -> Z then Z has two parents X and Y but only Y should be kept
    # Determinist case: minimize the number of parameters: take the minimal nb of levels
    # Quasi-determinist case: take the lowest entropy
    entropy_order_inc = (df_stats.H.values).argsort()
    # parent_order = compute_nb_levels(df) if is_strict else entropy_based_order
    A_df_forest = to_forest(A_df_noredundancy, entropy_order_inc)

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

    return A_df_forest


class Entropies(object):
    """ to do this could be easier to read with pyfields and default value factories """

    __slots__ = ('dataset', 'is_nparray', '_H', '_Hcond', '_Hcond_rel')

    def __init__(self, df_or_array):
        """

        :param df_or_array: a pandas dataframe or numpy array where the variables are the columns
        """
        self.dataset = df_or_array
        self.is_nparray = isinstance(df_or_array, np.ndarray)
        self._H = None          # type: Union[np.ndarray, pd.Series]
        self._Hcond = None      # type: Union[np.ndarray, pd.DataFrame]
        self._Hcond_rel = None  # type: Union[np.ndarray, pd.DataFrame]

    @property
    def varnames(self):
        if self.is_nparray:
            raise ValueError("Variable names not available")
        return list(self.dataset.columns)

    @property
    def H(self):
        """The entropies of all variables. a pandas Series if df was a pandas dataframe, else a 1D numpy array"""
        if self._H is None:
            # Using pyitlib to compute H (hopefully efficiently)
            self._H = drv.entropy(self.dataset.T)
            if not self.is_nparray:
                self._H = pd.Series(self._H, index=self.varnames)

            # basic sanity check: should all be positive
            assert np.all(self._H >= 0)
        return self._H

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
            self._Hcond = drv.entropy_conditional(self.dataset.T)

            # add the row/column headers
            if not self.is_nparray:
                self._Hcond = pd.DataFrame(self._Hcond, index=self.varnames, columns=self.varnames)

            # basic sanity check: should all be positive
            assert np.all(self._Hcond >= 0)
        return self._Hcond

    @property
    def Hcond_rel(self):
        """Relative conditional entropies of row vs. col. """
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
        edges = (df_stats.Hcond.values <= eps_absolute).T
    else:
        if eps_absolute is not None:
            raise ValueError("Only one of `eps_absolute` and `eps_relative` should be provided")

        # threshold on H(X|Y)/H(X)
        edges = (df_stats.Hcond_rel.values <= eps_relative).T

    # the diagonal should be false
    np.fill_diagonal(edges, False)

    # finally create the matrix
    A_df = pd.DataFrame(edges, index=df_stats.varnames, columns=df_stats.varnames)

    return A_df, df_stats


def remove_redundancies(A_df,
                        selection_order=None
                        ):
    """Cleans the arcs in A_df between redundant variables.

    A_df should be a dataframe where index = columns and indicate the node name.
    It should contain boolean values where True at (i,j) means there is a directed arc i -> j

    When there are redundant variables (arcs between i->j and j->i), only a single node is kept as the parent
    of all other redundant nodes.

    :param A_df: an adjacency matrix where A(i, j) indicates that there is an arc i -> j
    :param selection_order: if none (default) the first node in the list will be kept as representant for each
        redundancy class. Otherwise an alternate order (array of indices) can be provided.
    :return:
    """
    # work on a copy
    A_df = A_df.copy(deep=True)

    # init
    A = A_df.values
    n_vars = A_df.shape[0]
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

    return A_df


def to_forest(A_df, selection_order, inplace=False):
    """ Removes extra arcs in the adjacency matrix Z_df by only keeping the first parent in the given order

    :param A_df:
    :return:
    """
    if not inplace:
        A_df = A_df.copy()

    # nb_parents = A_df.sum(axis=0)
    # nodes_with_many_parents = np.where(nb_parents.values > 1)[0]

    # Probably the fastest: when the cumulative nb of parents is above 1 they need to be removed
    A_df[A_df.iloc[selection_order, :].cumsum(axis=0) > 1] = False

    # # Probably slower but easier to understand
    # # for each column we want to only keep one (or zero) row (parent) with True
    # for i in range(A_df.shape[1]):
    #     first_parent_idx = A_df.values[selection_order, i].argmax(axis=0)
    #     A_df.values[selection_order[first_parent_idx+1:], i] = False

    if not inplace:
        return A_df


# def to_forest(A_df, parent_scores=None, criterion_func=None):
#     """
#
#     :param A_df:
#     :param parent_scores:
#     :param criterion_func: if parent_scores is a
#     :return:
#     """
#     A_df = A_df.copy(deep=True)
#
#     if parent_scores is not None and criterion_func is not None:
#         raise ValueError("only one of `parent_scores` or `criterion_func` should be provided")
#
#     elif parent_scores is not None:
#         # for each child (column) we want to keep only a single parent
#         if parent_scores.shape != (A_df.shape[0], ):
#             raise ValueError("shape of `parent_scores` does not match")
#
#         A_df * parent_scores
#
#     elif criterion_func is not None:
#         # TODO use criterion_func(i, j) to get the score
#         raise NotImplementedError()
#
#     else:
#         raise ValueError("at least one of `parent_scores` or `criterion_func` should be provided")
#
#     return A_df


def compute_nb_levels(df):
    """

    :param df:
    :return:
    """
    df
