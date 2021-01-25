import numpy as np
import pandas as pd
import pytest

from qdscreen import qdeterscreen, Entropies, QDSSelector


def df_mix1():
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


@pytest.mark.parametrize("input_type", ["pandas",
                                        "numpy_unstructured",
                                        # "numpy_structured", "numpy_recarray"
                                        ])
def test_readme(input_type):
    """A test with a complete scenario of what is implemented so far"""

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
    df_stats = Entropies(data)

    # Sklearn use case
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
    # --strict
    ref = pd.DataFrame(data=np.zeros((nb_vars, nb_vars), dtype=bool), index=var_names, columns=var_names)
    ref.loc['U', 'V'] = True
    ref.loc['V', 'W'] = True
    ref.loc['X', 'Y'] = True
    qd_forest = qdeterscreen(data)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    pd.testing.assert_frame_equal(adj_mat, ref)

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
    qd_forest.fit(data_randcol)
    sel2 = qd_forest.select_qd_roots(data_randcol)
    if input_type == "pandas":
        pd.testing.assert_frame_equal(sel2, data_roots_df[new_roots_order])
    # elif input_type == "numpy_structured":
    #     sel2u = np.array(sel2.tolist(), dtype=object)
    #     np.testing.assert_array_equal(sel2u, data_roots_ar)
    else:
        np.testing.assert_array_equal(sel2, data_roots_ar)

    data2 = qd_forest.predict_qd_features_from_roots(sel2)
    if input_type == "pandas":
        pd.testing.assert_frame_equal(data2[data.columns], data)
    # elif input_type == "numpy_structured":
    #     np.testing.assert_array_equal(data2, data_ar)
    else:
        np.testing.assert_array_equal(data2, data_ar)

    # quasi
    ref.loc['X', 'Z'] = True
    # Z si threshold qd > 0.28 (absolu) ou 0.29 (relatif)
    qd_forest = qdeterscreen(data, absolute_eps=0.28)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    pd.testing.assert_frame_equal(adj_mat, ref)

    qd_forest = qdeterscreen(data, relative_eps=0.29)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        assert isinstance(qd_forest.parents, np.ndarray)
        assert isinstance(adj_mat, np.ndarray)
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    else:
        assert isinstance(qd_forest.parents, pd.DataFrame)
        assert list(qd_forest.parents.index) == var_names
    pd.testing.assert_frame_equal(adj_mat, ref)
