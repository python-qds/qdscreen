from copy import copy

import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv


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
            epsilon_absolute = 0
            is_absolute = True
        else:
            is_absolute = False
    else:
        is_absolute = True

    if is_absolute and epsilon_absolute < 0:
        raise ValueError("epsilon_absolute should be positive")
    elif not is_absolute and (epsilon_relative < 0 or epsilon_relative > 1):
        raise ValueError("epsilon_relative should be 0=<eps=<1")

    # (0) compute conditional entropies or relative conditional entropies
    H_df = compute_conditional_entropies(df, is_absolute)

    # (1) initial adjacency matrix (transpose because H is the other way round)
    A_df_orig = pd.DataFrame((H_df.values == 0).T, index=list(H_df.columns), columns=list(H_df.columns))

    # (2) identify redundancy
    if (is_absolute and epsilon_absolute == 0) or (not is_absolute and epsilon_relative == 0):
        A_df_noredundancy = identify_redundancy_strict(A_df_orig)
    else:
        # TODO implement
        A_df_noredundancy = identify_redundancy_quasi(A_df_orig)

    # (3) transform into forest: remove extra parents
    parent_scores = compute_nb_levels(df)
    A_df_forest = to_forest(A_df_noredundancy, parent_scores)

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

    return adjmat


def compute_conditional_entropies(df, is_absolute):
    """
    R function: MiMatrix

    :param is_absolute:
    :return: a conditional entropy matrix where element (i, j) is H(Xi|Xj)
    """
    # (0)init H
    # nb_vars = len(df.columns)
    # H = np.empty((nb_vars, nb_vars), dtype=float)

    # (1) for each column compute the counts per value

    # (2) for each (i, j) pair compute the counts

    # compute H efficiently (hopefully)
    H = drv.entropy_conditional(df.T)

    # add the row/column headers
    H_df = pd.DataFrame(H, index=list(df.columns), columns=list(df.columns))

    return H_df


def identify_redundancy_strict(A_df):
    """

    :param A_df:
    :return: an adjacency matrix where A(i, j) indicates that there is an arc i -> j
    """
    # work on a copy
    A_df = A_df.copy(deep=True)

    # init
    A = A_df.values
    n_vars = A_df.shape[0]

    # I contains the list of variable indices to go through. We can remove some in the loop
    I = np.ones((n_vars, ), dtype=bool)
    for i in range(n_vars):
        if I[i]:
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

    # restore the diagonal
    np.fill_diagonal(A, True)

    return A_df


def to_forest(A_df, parent_scores=None, criterion_func=None):
    """

    :param A_df:
    :param parent_scores:
    :param criterion_func: if parent_scores is a
    :return:
    """
    A_df = A_df.copy(deep=True)

    if parent_scores is not None and criterion_func is not None:
        raise ValueError("only one of `parent_scores` or `criterion_func` should be provided")

    elif parent_scores is not None:
        # for each child (column) we want to keep only a single parent
        if parent_scores.shape != (A_df.shape[0], ):
            raise ValueError("shape of `parent_scores` does not match")

        A_df * parent_scores

    elif criterion_func is not None:
        # TODO use criterion_func(i, j) to get the score
        raise NotImplementedError()

    else:
        raise ValueError("at least one of `parent_scores` or `criterion_func` should be provided")

    return A_df


def compute_nb_levels(df):
    """

    :param df:
    :return:
    """
    df
