# -*- coding: utf-8 -*-
# the above encoding declaration is needed to have non-ascii characters in this file (anywhere even in comments)
# from __future__ import unicode_literals  # no, since we want to match the return type of str() which is bytes in py2
import numpy as np
import pandas as pd
import pytest

from qdscreen import QDForest
from qdscreen.compat import PY2
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


def get_qd_forest1(is_np):
    """
    Created a forest with two trees 3->5 and 9->(1,7)

    :param is_np: if true
    :return:
    """
    adjmat_ar = adjmat = np.zeros((10, 10), dtype=bool)
    adjmat[1, 8] = True
    adjmat[3, 5] = True
    adjmat[9, 1] = True
    adjmat[9, 7] = True

    parents_ar = parents = -np.ones((10,),
                                    dtype=np.int64)  # indeed computing parents from adjmat with np.where returns this dtype
    parents[5] = 3
    parents[1] = 9
    parents[7] = 9
    parents[8] = 1

    roots_ar = roots = np.array([0, 2, 3, 4, 6, 9])
    roots_wc_ar = roots_wc = np.array([3, 9])

    if not is_np:
        varnames = list("abcdefghij")
        adjmat = pd.DataFrame(adjmat_ar, columns=varnames, index=varnames)
        parents = pd.DataFrame(parents_ar, index=varnames, columns=('idx',))
        parents['name'] = parents.index[parents['idx']].where(parents['idx'] >= 0, None)
        roots = np.array(varnames)[roots_ar]
        roots_wc = np.array(varnames)[roots_wc_ar]

    return adjmat, adjmat_ar, parents, parents_ar, roots, roots_ar, roots_wc, roots_wc_ar


@pytest.mark.parametrize("from_adjmat", [True, False], ids="from_adjmat={}".format)
@pytest.mark.parametrize("is_np", [True, False], ids="is_np={}".format)
def test_qd_forest(is_np, from_adjmat):
    """Tests that QDForest works correctly whether created from adj matrix or parents list"""

    adjmat, adjmat_ar, parents, parents_ar, roots, roots_ar, roots_wc, roots_wc_ar = get_qd_forest1(is_np)

    if from_adjmat:
        qd1 = QDForest(adjmat=adjmat)  # a forest created from the adj matrix
    else:
        qd1 = QDForest(parents=parents)  # a forest created from the parents coordinates

    # roots
    assert qd1.get_roots(names=False) == list(roots_ar)
    np.testing.assert_array_equal(qd1.indices_to_mask(roots_ar), qd1.roots_mask_ar)
    assert qd1.get_roots_with_children(names=False) == list(roots_wc_ar)
    if not is_np:
        pd.testing.assert_series_equal(qd1.indices_to_mask(roots), qd1.roots_mask)
        assert qd1.get_roots(names=True) == list(roots)
        assert qd1.get_roots_with_children(names=True) == list(roots_wc)
    else:
        with pytest.raises(ValueError):
            qd1.get_roots(names=True)
        with pytest.raises(ValueError):
            qd1.get_roots_with_children(names=True)
    # default
    assert qd1.get_roots() == qd1.get_roots(names=not is_np)
    assert qd1.get_roots_with_children() == qd1.get_roots_with_children(names=not is_np)

    # string representation of arcs
    # -- indices
    if from_adjmat:
        assert qd1.get_arcs_str_list(names=False) == ['1 -> 8', '3 -> 5', '9 -> 1', '9 -> 7']
    else:
        # make sure the adj matrix was not computed automatically, to test the right method
        assert qd1._adjmat is None
        # TODO order is not the same as above, see https://github.com/python-qds/qdscreen/issues/9
        assert qd1.get_arcs_str_list(names=False) == ['9 -> 1', '3 -> 5', '9 -> 7', '1 -> 8']
    # -- names
    if not is_np:
        if from_adjmat:
            assert qd1.get_arcs_str_list(names=True) == ['b -> i', 'd -> f', 'j -> b', 'j -> h']
        else:
            # make sure the adj matrix was not computed automatically, to test the right method
            assert qd1._adjmat is None
            # TODO order is not the same as above, see https://github.com/python-qds/qdscreen/issues/9
            assert qd1.get_arcs_str_list(names=True) == ['j -> b', 'd -> f', 'j -> h', 'b -> i']
    else:
        with pytest.raises(ValueError):
            qd1.get_arcs_str_list(names=True)
    # check the default value of `names`
    assert qd1.get_arcs_str_list() == qd1.get_arcs_str_list(names=not is_np)

    # equivalent adjmat and parents computation, to be sure
    np.testing.assert_array_equal(qd1.parents_indices_ar, parents_ar)
    if not is_np:
        pd.testing.assert_frame_equal(qd1.parents, parents)

    # equivalent adjmat computation, to be sure
    np.testing.assert_array_equal(qd1.adjmat_ar, adjmat_ar)
    if not is_np:
        pd.testing.assert_frame_equal(qd1.adjmat, adjmat)


@pytest.mark.parametrize("from_adjmat", [True, False])
@pytest.mark.parametrize("is_np", [True, False])
def test_qd_forest_str(is_np, from_adjmat):
    """Tests the string representation """

    adjmat, adjmat_ar, parents, parents_ar, roots, roots_ar, roots_wc, roots_wc_ar = get_qd_forest1(is_np)

    if from_adjmat:
        qd1 = QDForest(adjmat=adjmat)  # a forest created from the adj matrix
    else:
        qd1 = QDForest(parents=parents)  # a forest created from the parents coordinates

    # string representation
    compact_str = qd1.to_str(mode="compact")
    assert compact_str == "QDForest (10 vars = 6 roots + 4 determined by 2 of the roots)"

    roots_str = "0, 2, 3*, 4, 6, 9*" if is_np else "a, c, d*, e, g, j*"
    others_str = "1, 5, 7, 8" if is_np else "b, f, h, i"
    headers_str = qd1.to_str(mode="headers")
    assert headers_str == """QDForest (10 vars):
 - 6 roots (4+2*): %s
 - 4 other nodes: %s""" % (roots_str, others_str)
    # this should be the default
    assert qd1.to_str() == headers_str

    trees_str = "\n" + "\n".join(qd1.get_trees_str_list())
    if is_np:
        # note the u for python 2 as in main.py we use unicode literals to cope with those non-base chars
        assert trees_str == u"""
3
└─ 5

9
└─ 1
   └─ 8
└─ 7
"""
    else:
        # note the u for python 2 as in main.py we use unicode literals to cope with those non-base chars
        assert trees_str == u"""
d
└─ f

j
└─ b
   └─ i
└─ h
"""
    full_str = qd1.to_str(mode="full")
    assert full_str == headers_str + u"\n" + trees_str
    # this should be the default string representation if the nb vars is small enough
    if PY2:
        assert full_str.encode('utf-8') == str(qd1)
    else:
        assert full_str == str(qd1)


# def test_sklearn_compat():
#     """Trying to make sure that this compatibility code works: it does NOT with scikit-learn 0.22.2.post1 :("""
#     from qdscreen.compat import BaseEstimator
#     assert BaseEstimator()._more_tags()['requires_y'] is False
#     assert BaseEstimator()._get_tags()['requires_y'] is False
