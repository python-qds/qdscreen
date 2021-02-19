# API reference

In general, using `help(symbol)` is the recommended way to get the latest documentation. In addition, this page provides an overview of the various elements in this package.

## Main symbols

### `qd_screen`

```python
def qd_screen(X: Union[pd.DataFrame, np.ndarray],
              absolute_eps: float = None,
              relative_eps: float = None,
              keep_stats: bool = False
              ) -> QDForest
```

Finds the (quasi-)deterministic relationships (functional dependencies) between the variables in `X`, and returns a `QDForest` object representing the forest of (quasi-)deterministic trees. This object can then be used to fit a feature selection model or to learn a Bayesian Network structure.

By default only deterministic relationships are detected. Quasi-determinism can be enabled by setting either an threshold on conditional entropies (`absolute_eps`) or on relative conditional entropies (`relative_eps`). Only one of them should be set.

By default (`keep_stats=False`) the entropies tables are not preserved once the forest has been created. If you wish to keep them available, set `keep_stats=True`. The entropies tables are then available in the `<QDForest>.stats` attribute, and threshold analysis methods such as `<QDForest>.get_entropies_table(...)` and `<QDForest>.plot_increasing_entropies()` become available.

**Parameters:**

 * `X`: the dataset as a pandas DataFrame or a numpy array. Columns represent the features to compare.

 * `absolute_eps`: Absolute entropy threshold. Any feature `Y` that can be predicted from another feature `X` in a quasi-deterministic way, that is, where conditional entropy `H(Y|X) <= absolute_eps`, will be removed. The default value is `0` and corresponds to removing deterministic relationships only.

 * `relative_eps`: Relative entropy threshold. Any feature `Y` that can be predicted from another feature `X` in a quasi-deterministic way, that is, where relative conditional entropy `H(Y|X)/H(Y) <= relative_eps` (a value between `0` and `1`), will be removed. Only one of `absolute_eps` or `relative_eps` should be provided.

 * `keep_stats`: a boolean indicating if the various entropies tables computed in the process should be kept in memory in the resulting forest object (`<QDForest>.stats`), for further analysis. By default this is `False`. 

### `QDForest`

TODO

### `QDSelectorModel`

TODO

### `Entropies`

TODO

## scikit-learn symbols

### `QDSSelector`
