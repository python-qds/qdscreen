# Changelog

### 0.6.0 - sklearn api renames

 - Migrated CI/CD from Travis to Github Actions + `nox`.

 - `selector_skl` module renamed `sklearn` and `QDSSelector` renamed `QDScreen`. Fixes [#16](https://github.com/python-qds/qdscreen/issues/16)

### 0.5.0 - First public working release

Initial release with:

 * A main method `qd_screen` to get the (adjacency matrix of) the quasi-deterministic-forest, a `QDForest` object with string representation of arcs (Fixes [#8](https://github.com/python-qds/qdscreen/issues/8)).
 * Possibility to `keep_stats` so as to analyse the (conditional) entropies in order to define a "good" threshold. 
 * A method `<QDForest>.fit_selector_model(X)` to fit a `QDSelectorModel` feature selection model able to select relevant features and to predict missing ones. Fixes [#7](https://github.com/python-qds/qdscreen/issues/7)
 * Support for both pandas dataframes and numpy arrays as input. Fixes [#2](https://github.com/python-qds/qdscreen/issues/2)
 * A Scikit-learn compliant feature selector `QDSSelector`, providing the exact same functionality as above but compliant with scikit-learn `Pipeline`s. Fixes [#1](https://github.com/python-qds/qdscreen/issues/1)

Non-functional:

 * Travis continuous integration, generating documentation and deploying releases on PyPi
 * A package level `__version__` attribute. Fixes [#3](https://github.com/python-qds/qdscreen/issues/3)
 * Added `py.typed` for PEP561 compliance. Fixed [#4](https://github.com/python-qds/qdscreen/issues/4)
 * Initial `setup.py` and `setup.cfg`
