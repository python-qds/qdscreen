import numpy as np
import pandas as pd
import pytest

from qdscreen import qdeterscreen, Entropies


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
    # H(W|V) = 0     H(W|U) = 0  H(V|U) = 0              # U -> V -> W
    # H(Y|X) = H(X|Y) = 0                                # Y <-> X but X selected as representant as 1st appearing
    # H(Z|Y) = H(Z|X) = 0.275 AND H(Z|Y) / H(Z) = 0.284  # X -> Z if qd threshold > 0.19 (absolute) or 0.28 (relative)
    return df


@pytest.mark.parametrize("input_type", ["pandas", "numpy_unstructured", "numpy_structured", "numpy_recarray"])
def test_readme(input_type):
    """A test with a complete scenario of what is implemented so far"""

    if input_type == "pandas":
        data = df_mix1()
        var_names = list(data.columns)
    elif input_type == "numpy_unstructured":
        df_orig = df_mix1()
        var_names = list(df_orig.columns)
        data = df_orig.to_numpy(copy=True)
    elif input_type == "numpy_structured":
        df_orig = df_mix1()
        var_names = list(df_orig.columns)
        data = df_orig.to_records()
        data = data.view(data.dtype.fields or data.dtype, np.ndarray)
    elif input_type == "numpy_recarray":
        df_orig = df_mix1()
        var_names = list(df_orig.columns)
        data = df_orig.to_records()
    else:
        raise ValueError(input_type)

    nb_vars = len(var_names)

    # check the stats, for debug
    df_stats = Entropies(data)

    # strict
    ref = pd.DataFrame(data=np.zeros((nb_vars, nb_vars), dtype=bool), index=var_names, columns=var_names)
    ref.loc['U', 'V'] = True
    ref.loc['V', 'W'] = True
    ref.loc['X', 'Y'] = True
    qd_forest = qdeterscreen(data)
    adj_mat = qd_forest.adjmat
    if input_type != "pandas":
        adj_mat = pd.DataFrame(adj_mat, columns=var_names, index=var_names)
    pd.testing.assert_frame_equal(adj_mat, ref)

    # TODO check that this works now - dataframe backed by sparse array
    # sel = QDSSelector()
    # sel_data = sel.fit_transform(data.values)
    # data2 = sel.inverse_transform(sel_data)

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
        assert isinstance(qd_forest.parents, pd.Series)
        assert list(qd_forest.parents.columns) == var_names
    pd.testing.assert_frame_equal(adj_mat, ref)
