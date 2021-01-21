import numpy as np
import pandas as pd
import pytest

from qdscreen import QDForest
from qdscreen.main import get_adjacency_matrix, remove_redundancies


def df_strict1():
    """ A dataframe with two equivalent redundant variables X and Y """
    df = pd.DataFrame({
        'X': ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
    })
    # 'Y':   ["b", "b", "c", "c", "b", "a", "a", "b", "c", "a"],
    df['Y'] = df['X'].replace("c", "d").replace("b", "c").replace("a", "b").replace("d", "a")
    return df


def test_adjacency_strict():
    """ Tests that `get_adjacency_matrix` works as expected in front of a dataset with redundancy """

    # Compute the adjacency matrix
    adj_df, df_stats = get_adjacency_matrix(df_strict1())
    pd.testing.assert_frame_equal(adj_df, pd.DataFrame(data=[
        [False, True],
        [True, False]
    ], index=['X', 'Y'], columns=['X', 'Y']))


def quasi_df1():
    # in that matrix the H(X|Y) = H(Y|X) = 0.324511 and the relative H(X|Y)/H(X) = H(Y|X)/H(Y) = 0.20657
    return pd.DataFrame({
        'X': ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
        'Y': ["a", "b", "c", "c", "b", "a", "a", "b", "c", "a"],
    })


def test_adjacency_invalid_thresholds():
    # an error is raised if both are provided, even if both are zero
    with pytest.raises(ValueError):
        get_adjacency_matrix(pd.DataFrame(), eps_absolute=0., eps_relative=0.)


@pytest.mark.parametrize("eps_absolute, eps_relative", [
    (None, None),  # strict mode
    (0.2, None),   # too low absolute
    (None, 0.15)   # too low relative
])
def test_adjacency_quasi_low_threshold(eps_absolute, eps_relative):
    """ Tests that `get_adjacency_matrix` works as expected in quasi mode """

    # in that matrix the H(X|Y) = H(Y|X) = 0.324511 and the relative H(X|Y)/H(X) = H(Y|X)/H(Y) = 0.20657
    # strict mode or with too low threshold: no arc is detected
    adj_df, df_stats = get_adjacency_matrix(quasi_df1(), eps_absolute=eps_absolute, eps_relative=eps_relative)
    pd.testing.assert_frame_equal(adj_df, pd.DataFrame(data=[
        [False, False],
        [False, False]
    ], index=['X', 'Y'], columns=['X', 'Y']))


@pytest.mark.parametrize("eps_absolute, eps_relative", [
    (0.33, None),  # high enough absolute
    (None, 0.21)   # high enough relative
])
def test_adjacency_quasi_high_threshold(eps_absolute, eps_relative):
    """ Tests that `get_adjacency_matrix` works as expected in quasi mode """

    # in that matrix the H(X|Y) = H(Y|X) = 0.324511 and the relative H(X|Y)/H(X) = H(Y|X)/H(Y) = 0.20657
    # strict mode or with too low threshold: no arc is detected
    adj_df, df_stats = get_adjacency_matrix(quasi_df1(), eps_absolute=eps_absolute, eps_relative=eps_relative)
    pd.testing.assert_frame_equal(adj_df, pd.DataFrame(data=[
        [False, True],
        [True, False]
    ], index=['X', 'Y'], columns=['X', 'Y']))


def test_remove_redundancies():
    """ Tests that the redundancies removal routine works as expected """

    # an adjacency matrix with two redundant nodes X and Y
    adj_df = pd.DataFrame(data=[
        [False, True],
        [True, False]
    ], index=['X', 'Y'], columns=['X', 'Y'])

    # clean the redundancies with natural order: X is the representative of the class
    adj_df_clean1 = remove_redundancies(adj_df)
    pd.testing.assert_frame_equal(adj_df_clean1, pd.DataFrame(data=[
        [False, True],
        [False, False]
    ], index=['X', 'Y'], columns=['X', 'Y']))

    # clean the redundancies with reverse order: Y is the representative of the class
    adj_df_clean2 = remove_redundancies(adj_df, selection_order=[1, 0])
    pd.testing.assert_frame_equal(adj_df_clean2, pd.DataFrame(data=[
        [False, False],
        [True, False]
    ], index=['X', 'Y'], columns=['X', 'Y']))


# def test_identify_redundancy_quasi():
#     df = pd.DataFrame({
#         'U': ["a", "b", "d", "a", "b", "c", "a", "b", "d", "c"],
#     })
#     df['V'] = df['U'].replace("d", "c")  # d -> c
#     df['W'] = df['V'].replace("c", "b")  # c -> b
#
#     adj_df = get_adjacency_matrix(df)
#     df2 = identify_redundancy(adj_df)


@pytest.mark.parametrize("is_np", [True, False])
def test_qd_forest(is_np):
    """Tests that QDForest works correctly whether created from adj matrix or parents list"""

    # reference matrices: a forest with two trees 3->5 and 9->(1,7)
    adjmat_ar = adjmat = np.zeros((10, 10), dtype=bool)
    adjmat[3, 5] = True
    adjmat[9, 1] = True
    adjmat[9, 7] = True
    parents_ar = parents = -np.ones((10,), dtype=np.int64)  # indeed computing parents from adjmat with np.where returns this dtype
    parents[5] = 3
    parents[1] = 9
    parents[7] = 9
    if not is_np:
        varnames = list("abcdefghij")
        adjmat = pd.DataFrame(adjmat_ar, columns=varnames, index=varnames)
        parents = pd.Series(parents_ar, index=varnames)

    # a forest created from the adj matrix
    qd1 = QDForest(adjmat=adjmat)
    np.testing.assert_array_equal(qd1.parents_ar, parents_ar)
    if not is_np:
        pd.testing.assert_series_equal(qd1.parents, parents)

    # a forest created from the parents coordinates
    qd2 = QDForest(parents=parents)
    np.testing.assert_array_equal(qd2.adjmat_ar, adjmat_ar)
    if not is_np:
        pd.testing.assert_frame_equal(qd1.adjmat, adjmat)
