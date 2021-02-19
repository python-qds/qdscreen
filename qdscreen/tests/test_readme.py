# -*- coding: utf-8 -*-
# the above encoding declaration is needed to have non-ascii characters in this file (anywhere even in comments)
# from __future__ import unicode_literals  # no, since we want to match the return type of str() which is bytes in py2
import sys

import numpy as np
import pandas as pd
import pytest

from qdscreen import qd_screen


try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


PY2 = sys.version_info < (3,)
IMGS_FOLDER = Path(__file__).parent.parent.parent / "docs" / "imgs"


def df_mix1():
    """The dataset for reuse"""
    df = pd.DataFrame({
        'U': ["a", "b", "d", "a", "b", "c", "a", "b", "d", "c"],
        'V': ["a", "b", "c", "a", "b", "c", "a", "b", "c", "c"],
        'W': ["a", "b", "b", "a", "b", "b", "a", "b", "b", "b"],
        'X': ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
        'Y': ["b", "b", "c", "c", "b", "a", "a", "b", "c", "a"],
        'Z': ["a", "a", "b", "a", "a", "b", "b", "a", "a", "b"]  # if the 3d element were an "a", X->Z and Y->Z
    })

    # In this data: the following relationships hold:
    # H(W|V) = 0     H(W|U) = 0  H(V|U) = 0              # U -> V -> W
    # H(Y|X) = H(X|Y) = 0                                # Y <-> X but X selected as representant as 1st appearing
    # H(Z|Y) = H(Z|X) = 0.275 AND H(Z|Y) / H(Z) = 0.284  # X -> Z if qd threshold > 0.19 (absolute) or 0.28 (relative)
    return df


def test_encoding(capsys):
    """A small test to be sure we can compare the tree text characters correctly"""
    print("└─")
    captured = capsys.readouterr()
    with capsys.disabled():
        # here this is unicode match
        assert captured.out == u"└─\n"

        # here this is bytes match
        from .encoding_ref_help import Foo
        assert "└─ab\n" == str(Foo())
        assert "└─ab\n" == repr(Foo())


def test_readme_simple(capsys):
    """The exact scenario showed in the readme"""

    df = df_mix1()

    # detect strict deterministic relationships
    qd_forest = qd_screen(df)
    print(qd_forest)
    captured = capsys.readouterr()
    with capsys.disabled():
        # note the u for unicode string in py2, see test_encoding
        assert "\n" + captured.out == u"""
QDForest (6 vars):
 - 3 roots (1+2*): U*, X*, Z
 - 3 other nodes: V, W, Y

U
└─ V
   └─ W

X
└─ Y

"""
    print("Columns in df: %s" % list(df.columns))

    # fit a feature selection model
    feat_selector = qd_forest.fit_selector_model(df)

    # use it to filter...
    only_important_features = feat_selector.remove_qd(df)
    print("Columns in only_important_features: %s" % list(only_important_features.columns))

    # or to restore/predict
    restored_full_df = feat_selector.predict_qd(only_important_features)
    print("Columns in restored_full_df: %s" % list(restored_full_df.columns))

    # note that the order of columns differs frmo origin
    pd.testing.assert_frame_equal(df, restored_full_df[df.columns])

    captured = capsys.readouterr()
    with capsys.disabled():
        assert "\n" + captured.out == """
Columns in df: ['U', 'V', 'W', 'X', 'Y', 'Z']
Columns in only_important_features: ['U', 'X', 'Z']
Columns in restored_full_df: ['U', 'X', 'Z', 'V', 'W', 'Y']
"""


def test_readme_quasi(capsys):
    df = df_mix1()

    # keep stats at the end of the screening
    qd_forest = qd_screen(df, keep_stats=True)
    print(qd_forest.stats)

    captured = capsys.readouterr()
    with capsys.disabled():
        # print(captured.out)
        assert "\n" + captured.out == """
Statistics computed for dataset:
   U  V  W  X  Y  Z
0  a  a  a  a  b  a
1  b  b  b  a  b  a
...(10 rows)

Entropies (H):
U    1.970951
V    1.570951
W    0.881291
X    1.570951
Y    1.570951
Z    0.970951
dtype: float64

Conditional entropies (Hcond = H(row|col)):
          U         V         W         X         Y         Z
U  0.000000  0.400000  1.089660  0.875489  0.875489  1.475489
V  0.000000  0.000000  0.689660  0.875489  0.875489  1.200000
W  0.000000  0.000000  0.000000  0.875489  0.875489  0.875489
X  0.475489  0.875489  1.565148  0.000000  0.000000  0.875489
Y  0.475489  0.875489  1.565148  0.000000  0.000000  0.875489
Z  0.475489  0.600000  0.965148  0.275489  0.275489  0.000000

Relative conditional entropies (Hcond_rel = H(row|col)/H(row)):
          U         V         W         X         Y         Z
U  0.000000  0.202948  0.552860  0.444196  0.444196  0.748618
V  0.000000  0.000000  0.439008  0.557299  0.557299  0.763869
W  0.000000  0.000000  0.000000  0.993416  0.993416  0.993416
X  0.302676  0.557299  0.996307  0.000000  0.000000  0.557299
Y  0.302676  0.557299  0.996307  0.000000  0.000000  0.557299
Z  0.489715  0.617951  0.994024  0.283731  0.283731  0.000000

"""
    qd_forest2 = qd_screen(df, relative_eps=0.29)
    print(qd_forest2)
    captured = capsys.readouterr()
    with capsys.disabled():
        # print(captured.out)
        # note the u for unicode string in py2, see test_encoding
        assert "\n" + captured.out == u"""
QDForest (6 vars):
 - 2 roots (0+2*): U*, X*
 - 4 other nodes: V, W, Y, Z

U
└─ V
   └─ W

X
└─ Y
└─ Z

"""
    ce_df = qd_forest.get_entropies_table(from_to=False, sort_by="rel_cond_entropy")
    print(ce_df.head(10))
    captured = capsys.readouterr()
    with capsys.disabled():
        # print(captured.out)
        assert "\n" + captured.out == """
      cond_entropy  rel_cond_entropy
arc                                 
U->V      0.000000          0.000000
U->W      0.000000          0.000000
V->W      0.000000          0.000000
Y->X      0.000000          0.000000
X->Y      0.000000          0.000000
V->U      0.400000          0.202948
Y->Z      0.275489          0.283731
X->Z      0.275489          0.283731
U->X      0.475489          0.302676
U->Y      0.475489          0.302676
"""
    if not PY2:
        # we have issues on travis CI with matplotlib on PY2: skip
        import matplotlib.pyplot as plt
        qd_forest.plot_increasing_entropies()
        fig = plt.gcf()
        fig.savefig(str(IMGS_FOLDER / "increasing_entropies.png"))
        plt.close("all")


@pytest.mark.parametrize("typ", ['int', 'str', 'mixed'])
def test_readme_skl(typ):
    from qdscreen.selector_skl import QDSSelector

    if typ in ('int', 'str'):
        X = [[0, 2, 0, 3],
             [0, 1, 4, 3],
             [0, 1, 1, 3]]
        if typ == 'str':
            X = np.array(X).astype(str)

    elif typ == 'mixed':
        # X = [['0', 2, '0', 3],
        #      ['0', 1, '4', 3],
        #      ['0', 1, '1', 3]]
        # X = np.array(X)
        # assert X.dtype == '<U1' #str
        pytest.skip("Mixed dtypes do not exist in numpy unstructured arrays")
    else:
        raise ValueError()

    selector = QDSSelector()
    X2 = selector.fit_transform(X)
    expected_res = [[0], [4], [1]] if typ == 'int' else [['0'], ['4'], ['1']]
    np.testing.assert_array_equal(X2, np.array(expected_res))


@pytest.mark.parametrize("input_type", ["pandas",
                                        "numpy_unstructured",
                                        # "numpy_structured", "numpy_recarray"
                                        ])
def test_readme(input_type):
    """All variants that exist around the nominal scenario from the readme"""

    if input_type == "pandas":
        df_orig = data = df_mix1()
        var_names = list(data.columns)
        data_ar = data.values
    elif input_type == "numpy_unstructured":
        df_orig = df_mix1()
        var_names = list(df_orig.columns)
        data = df_orig.to_numpy(copy=True)
        data_ar = data
    # elif input_type == "numpy_structured":
    #     df_orig = df_mix1()
    #     var_names = list(df_orig.columns)
    #     data = df_orig.to_records()
    #     data = data.view(data.dtype.fields or data.dtype, np.ndarray)[var_names].copy()
    #     data_ar = data
    # elif input_type == "numpy_recarray":
    #     df_orig = df_mix1()
    #     var_names = list(df_orig.columns)
    #     data = df_orig.to_records()[var_names].copy()
    #     data_ar = data
    #     data.dtype.names = tuple(range(len(var_names)))
    else:
        raise ValueError(input_type)

    # ref roots arrays
    data_roots_df = df_orig[['U', 'X', 'Z']]
    data_roots_ar = data_roots_df.values
    data_roots_quasi_df = df_orig[['U', 'X']]
    data_roots_quasi_ar = data_roots_quasi_df.values

    nb_vars = len(var_names)

    # check the stats, for debug
    # df_stats = Entropies(data)

    # Sklearn use case
    from qdscreen.selector_skl import QDSSelector
    if input_type not in ("numpy_structured", "numpy_recarray"):
        # --strict
        sel = QDSSelector()
        sel_data = sel.fit_transform(data_ar)
        np.testing.assert_array_equal(sel_data, data_roots_ar)
        data2 = sel.inverse_transform(sel_data)
        np.testing.assert_array_equal(data2, data_ar)
        # --quasi absolute - Z should become a child of X
        sel = QDSSelector(absolute_eps=0.28)
        sel_data = sel.fit_transform(data_ar)
        np.testing.assert_array_equal(sel_data, data_roots_quasi_ar)
        data2 = sel.inverse_transform(sel_data)
        # the prediction is incorrect since this is "quasi". Fix the single element that differs from input
        assert data2[2, 5] == "a"
        data2[2, 5] = "b"
        np.testing.assert_array_equal(data2, data_ar)

        # --quasi relative - Z should become a child of X
        sel = QDSSelector(relative_eps=0.29)
        sel_data = sel.fit_transform(data_ar)
        np.testing.assert_array_equal(sel_data, data_roots_quasi_ar)
        data2 = sel.inverse_transform(sel_data)
        # the prediction is incorrect since this is "quasi". Fix the single element that differs from input
        assert data2[2, 5] == "a"
        data2[2, 5] = "b"
        np.testing.assert_array_equal(data2, data_ar)

    # qdscreen full use case
    # (1) strict
    # -- (a) forest
    ref = pd.DataFrame(data=np.zeros((nb_vars, nb_vars), dtype=bool), index=var_names, columns=var_names)
    ref.loc['U', 'V'] = True
    ref.loc['V', 'W'] = True
    ref.loc['X', 'Y'] = True
    qd_forest = qd_screen(data)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    pd.testing.assert_frame_equal(adj_mat, ref)
    # -- (b) feature selector
    if input_type == "pandas":
        # Note: this should be also tested with numpy structured & records input type
        # we can test that randomized column order still works
        new_cols_order = np.array(data.columns)
        np.random.shuffle(new_cols_order)
        new_roots_order = [c for c in new_cols_order if c in data_roots_df.columns]
        data_randcol = data[new_cols_order].copy()
    else:
        # numpy: index has meaning, no randomization possible
        data_randcol = data
    selector_model = qd_forest.fit_selector_model(data_randcol)
    sel2 = selector_model.remove_qd(data_randcol)
    if input_type == "pandas":
        pd.testing.assert_frame_equal(sel2, data_roots_df[new_roots_order])
    # elif input_type == "numpy_structured":
    #     sel2u = np.array(sel2.tolist(), dtype=object)
    #     np.testing.assert_array_equal(sel2u, data_roots_ar)
    else:
        np.testing.assert_array_equal(sel2, data_roots_ar)
    data2 = selector_model.predict_qd(sel2)
    if input_type == "pandas":
        pd.testing.assert_frame_equal(data2[data.columns], data)
    # elif input_type == "numpy_structured":
    #     np.testing.assert_array_equal(data2, data_ar)
    else:
        np.testing.assert_array_equal(data2, data_ar)

    # (2) quasi
    # -- (a) forest
    ref.loc['X', 'Z'] = True
    # Z si threshold qd > 0.28 (absolu) ou 0.29 (relatif)
    qd_forest = qd_screen(data, absolute_eps=0.28)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    pd.testing.assert_frame_equal(adj_mat, ref)

    qd_forest = qd_screen(data, relative_eps=0.29)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        assert isinstance(qd_forest.parents, np.ndarray)
        assert isinstance(adj_mat, np.ndarray)
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    else:
        assert isinstance(qd_forest.parents, pd.DataFrame)
        assert list(qd_forest.parents.index) == var_names
    pd.testing.assert_frame_equal(adj_mat, ref)

    # -- (b) feature selector

    selector_model = qd_forest.fit_selector_model(data_randcol)
    sel2 = selector_model.remove_qd(data_randcol)
    if input_type == "pandas":
        new_roots_order_quasi = [r for r in new_roots_order if r != 'Z']
        pd.testing.assert_frame_equal(sel2, data_roots_df[new_roots_order_quasi])
    # elif input_type == "numpy_structured":
    #     sel2u = np.array(sel2.tolist(), dtype=object)
    #     np.testing.assert_array_equal(sel2u, data_roots_ar)
    else:
        np.testing.assert_array_equal(sel2, data_roots_quasi_ar)
    data2 = selector_model.predict_qd(sel2)
    if input_type == "pandas":
        # the prediction is incorrect since this is "quasi". Fix the single element that differs from input
        assert data2.loc[2, "Z"] == "a"
        data2.loc[2, "Z"] = "b"
        pd.testing.assert_frame_equal(data2[data.columns], data)
    # elif input_type == "numpy_structured":
    #     np.testing.assert_array_equal(data2, data_ar)
    else:
        # the prediction is incorrect since this is "quasi". Fix the single element that differs from input
        assert data2[2, 5] == "a"
        data2[2, 5] = "b"
        np.testing.assert_array_equal(data2, data_ar)
