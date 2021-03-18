import numpy as np
import pandas as pd


def patch_pandas_if_needed():
    try:
        pd.DataFrame.to_numpy
    except AttributeError:
        def to_numpy(self, dtype=None, copy=False):
            """
            Convert the DataFrame to a NumPy array.

            .. versionadded:: 0.24.0

            By default, the dtype of the returned array will be the common NumPy
            dtype of all types in the DataFrame. For example, if the dtypes are
            ``float16`` and ``float32``, the results dtype will be ``float32``.
            This may require copying data and coercing values, which may be
            expensive.

            Parameters
            ----------
            dtype : str or numpy.dtype, optional
                The dtype to pass to :meth:`numpy.asarray`
            copy : bool, default False
                Whether to ensure that the returned value is a not a view on
                another array. Note that ``copy=False`` does not *ensure* that
                ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
                a copy is made, even if not strictly necessary.

            Returns
            -------
            numpy.ndarray

            See Also
            --------
            Series.to_numpy : Similar method for Series.

            Examples
            --------
            >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
            array([[1, 3],
                   [2, 4]])

            With heterogenous data, the lowest common type will have to
            be used.

            >>> df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
            >>> df.to_numpy()
            array([[1. , 3. ],
                   [2. , 4.5]])

            For a mix of numeric and non-numeric types, the output array will
            have object dtype.

            >>> df['C'] = pd.date_range('2000', periods=2)
            >>> df.to_numpy()
            array([[1, 3.0, Timestamp('2000-01-01 00:00:00')],
                   [2, 4.5, Timestamp('2000-01-02 00:00:00')]], dtype=object)
            """
            result = np.array(self.values, dtype=dtype, copy=copy)
            return result

        pd.DataFrame.to_numpy = to_numpy

    else:
        pass
