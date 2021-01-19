import numpy as np
import pandas as pd
import pytest

from qdscreen.main import get_adjacency_matrix, remove_redundancies, Entropies
from qdscreen import qdeterscreen


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


def df_mix1():
    df = pd.DataFrame({
        'U': ["a", "b", "d", "a", "b", "c", "a", "b", "d", "c"],
        'V': ["a", "b", "c", "a", "b", "c", "a", "b", "c", "c"],
        'W': ["a", "b", "b", "a", "b", "b", "a", "b", "b", "b"],
        'X': ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
        'Y': ["b", "b", "c", "c", "b", "a", "a", "b", "c", "a"],
        'Z': ["a", "a", "b", "a", "a", "b", "b", "a", "a", "b"]
    })

    # In this data: the following relationships hold:
    # H(W|V) = 0     H(W|U) = 0  H(V|U) = 0                   # U -> V -> W
    # H(Y|X) = H(X|Y) = 0             # Y <-> X
    # H(Z|Y) = H(Z|X) = 0.275 AND H(Z|Y) / H(Z) = 0.284  # X (et pas Y car il n'est pas représentant) -> Z si threshold qd > 0.19 (absolu) ou 0.28 (relatif)
    return df


def test_walkthrough():

    # TODO check
    #    https://github.com/pgmpy/pgmpy
    #    https://pomegranate.readthedocs.io/en/latest/index.html
    #    http://bayespy.org/
    #    more generic: https://pyjags.readthedocs.io/en/latest/ and http://edwardlib.org/
    #
    # TODO the best is maybe   https://github.com/ncullen93/pyGOBN

    df = df_mix1()
    var_names = list(df.columns)
    nb_vars = len(var_names)

    df_stats = Entropies(df)

    # strict
    adj_mat = qdeterscreen(df)
    ref = pd.DataFrame(data=np.zeros((nb_vars, nb_vars), dtype=bool), index=var_names, columns=var_names)
    ref.loc['U', 'V'] = True
    ref.loc['V', 'W'] = True
    ref.loc['X', 'Y'] = True
    pd.testing.assert_frame_equal(adj_mat, ref)


    # quasi
    ref.loc['X', 'Z'] = True
    # Z si threshold qd > 0.28 (absolu) ou 0.29 (relatif)
    adj_mat = qdeterscreen(df, epsilon_absolute=0.28)
    pd.testing.assert_frame_equal(adj_mat, ref)

    adj_mat = qdeterscreen(df, epsilon_relative=0.29)
    pd.testing.assert_frame_equal(adj_mat, ref)
