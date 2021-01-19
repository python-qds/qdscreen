import numpy as np
import pandas as pd

from qdscreen.main import Entropies
from qdscreen import qdeterscreen


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


def test_readme():
    """A test with a complete scenario of what is implemented so far"""

    df = df_mix1()
    var_names = list(df.columns)
    nb_vars = len(var_names)

    # check the stats, for debug
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
