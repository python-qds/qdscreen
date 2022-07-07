# qdscreen

*Remove redundancy in your categorical variables and increase your models performance.*

[![Python versions](https://img.shields.io/pypi/pyversions/qdscreen.svg)](https://pypi.python.org/pypi/qdscreen/) [![Build Status](https://github.com/python-qds/qdscreen/actions/workflows/base.yml/badge.svg)](https://github.com/python-qds/qdscreen/actions/workflows/base.yml) [![Tests Status](./reports/junit/junit-badge.svg?dummy=8484744)](./reports/junit/report.html) [![Coverage Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](./reports/coverage/index.html) [![codecov](https://codecov.io/gh/smarie/python-odsclient/branch/main/graph/badge.svg)](https://codecov.io/gh/smarie/python-odsclient) [![Flake8 Status](./reports/flake8/flake8-badge.svg?dummy=8484744)](./reports/flake8/index.html)

[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://python-qds.github.io/qdscreen/) [![PyPI](https://img.shields.io/pypi/v/qdscreen.svg)](https://pypi.python.org/pypi/qdscreen/) [![Downloads](https://pepy.tech/badge/qdscreen)](https://pepy.tech/project/qdscreen) [![Downloads per week](https://pepy.tech/badge/qdscreen/week)](https://pepy.tech/project/qdscreen) [![GitHub stars](https://img.shields.io/github/stars/python-qds/qdscreen.svg)](https://github.com/python-qds/qdscreen/stargazers)

`qdscreen` provides a python implementation of the Quasi-determinism screening algorithm (also known as `qds-BNSL`) from T.Rahier's PhD thesis, 2018.

Most data scientists are familiar with the concept of *correlation* between continuous variables. This concept extends to categorical variables, and is known as *functional dependency* in the field of relational databases mining. We also name it *determinism* in the context of Machine Learning and Statistics, to indicate that when a random variable `X` is known then the value of another variable `Y` is determined with absolute certainty. *"Quasi-"*determinism is an extension of this concept to handle noise or extremely rare cases in data.

`qdscreen` is able to detect and remove (quasi-)deterministic relationships in a dataset:

 - either as a preprocessing step in any general-purpose data science pipeline
   
 - or as an accelerator of a Bayesian Network Structure Learning method such as [`pyGOBN`](https://github.com/ncullen93/pyGOBN)


## Installing

```bash
> pip install qdscreen
```

## Usage

### 1. Remove correlated variables

See [this example](./generated/gallery/1_remove_correlated_vars_demo.md).

### 2. Learn a Bayesian Network structure

TODO see [#6](https://github.com/python-qds/qdscreen/issues/6).


## Main features / benefits

 * A feature selection algorithm able to eliminate quasi-deterministic relationships
   
    - a base version compliant with numpy and pandas datasets
    - a scikit-learn compliant version (numpy only)

 * An accelerator for Bayesian Network Structure Learning tasks


## See Also

 - Bayesian Network libraries in python: 
   
    - [`pyGOBN`](https://github.com/ncullen93/pyGOBN) (MIT license)
    - [`pgmpy`](https://github.com/pgmpy/pgmpy) (MIT license)
    - [`pomegranate`](https://pomegranate.readthedocs.io/en/latest/index.html) (MIT license)
    - [`bayespy`](http://bayespy.org/) (MIT license)

 - Functional dependencies libraries in python:

    - [`fd_miner`](https://github.com/anonexp/fd_miner), an algorithm that was used in [this paper](https://hal.archives-ouvertes.fr/hal-01856516/document). The repository contains a list of reference datasets too.
    - [`FDTool`](https://github.com/USEPA/FDTool) a python 2 algorithm to mine for functional dependencies, equivalences and candidate keys. From [this paper](https://f1000research.com/articles/7-1667).
    - [`functional-dependencies`](https://github.com/amrith/functional-dependencies)
    - [`functional-dependency-finder`](https://github.com/gustavclausen/functional-dependency-finder) connects to a MySQL db and finds functional dependencies.

 - Other libs for probabilistic inference:

    - [`pyjags`](https://github.com/michaelnowotny/pyjags) (GPLv2 license)
    - [`edward`](http://edwardlib.org/) (Apache License, Version 2.0)

 - Stackoverflow discussions:

    - [detecting normal forms](https://stackoverflow.com/questions/2157531/python-code-for-determining-which-normal-form-tabular-data-is-in)
    - [canonical cover](https://stackoverflow.com/questions/2822809/program-to-find-canonical-cover-or-minimum-number-of-functional-dependencies)


### Others

*Do you like this library ? You might also like [smarie's other python libraries](https://github.com/smarie/OVERVIEW#python)* 

## Want to contribute ?

Details on the github page: [https://github.com/python-qds/qdscreen](https://github.com/python-qds/qdscreen)
