import pandas as pd

from qdsbnsl import qdeterscreen


def test_walkthrough():

    # TODO check
    #    https://github.com/pgmpy/pgmpy
    #    https://pomegranate.readthedocs.io/en/latest/index.html
    #    http://bayespy.org/
    #    more generic: https://pyjags.readthedocs.io/en/latest/ and http://edwardlib.org/
    #
    # TODO the best is maybe   https://github.com/ncullen93/pyGOBN

    df = pd.DataFrame({
        'V': ["a", "b", "c", "a", "b", "c", "a", "b", "c", "c"],
        'W': ["a", "b", "b", "a", "b", "b", "a", "b", "b", "b"],
        'X': ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
        'Y': ["b", "b", "c", "c", "b", "a", "a", "b", "c", "a"],
        'Z': ["a", "a", "b", "a", "a", "b", "b", "a", "a", "b"]
    })

    # In this data: the following relationships hold:
    # H(W|V) = 0
    # H(Y|X) = H(X|Y) = 0
    # H(Z|Y) = H(Z|X) = 0.19 AND H(Z|Y) / H(Z) = 0.28

    adj_mat = qdeterscreen(df)
