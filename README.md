# qdscreen

*Remove redundancy in your categorical variables and increase your models performance.*

[![Python versions](https://img.shields.io/pypi/pyversions/qdscreen.svg)](https://pypi.python.org/pypi/qdscreen/) [![Build Status](https://travis-ci.com/python-qds/qdscreen.svg?branch=main)](https://travis-ci.com/github/python-qds/qdscreen) [![Tests Status](https://python-qds.github.io/qdscreen/junit/junit-badge.svg?dummy=8484744)](https://python-qds.github.io/qdscreen/junit/report.html) [![codecov](https://codecov.io/gh/python-qds/qdscreen/branch/master/graph/badge.svg)](https://codecov.io/gh/python-qds/qdscreen)

[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://python-qds.github.io/qdscreen/) [![PyPI](https://img.shields.io/pypi/v/qdscreen.svg)](https://pypi.python.org/pypi/qdscreen/) [![Downloads](https://pepy.tech/badge/qdscreen)](https://pepy.tech/project/qdscreen) [![Downloads per week](https://pepy.tech/badge/qdscreen/week)](https://pepy.tech/project/qdscreen) [![GitHub stars](https://img.shields.io/github/stars/python-qds/qdscreen.svg)](https://github.com/python-qds/qdscreen/stargazers)

`qdscreen` provides a python implementation of the Quasi-determinism screening algorithm (also known as `qds-BNSL`) from T.Rahier's PhD thesis, 2018.
**This is the readme for developers.** The documentation for users is available here: [https://python-qds.github.io/qdscreen/](https://python-qds.github.io/qdscreen/)

## Want to contribute ?

Contributions are welcome ! Simply fork this project on github, commit your contributions, and create pull requests.

Here is a non-exhaustive list of interesting open topics: [https://github.com/python-qds/qdscreen/issues](https://github.com/python-qds/qdscreen/issues)

## Running the tests

This project uses `pytest`.

```bash
pytest
```

## Packaging

This project uses `setuptools_scm` to synchronise the version number. Therefore the following command should be used for development snapshots as well as official releases: 

```bash
python setup.py egg_info bdist_wheel rotate -m.whl -k3
```

## Generating the documentation page

This project uses `mkdocs` to generate its documentation page. Therefore building a local copy of the doc page may be done using:

```bash
mkdocs build -f docs/mkdocs.yml
```

You can even serve it with automatic sync in case of modifications:

```bash
mkdocs serve -f docs/mkdocs.yml
```

## Generating the test reports

The following commands generate the html test report and the associated badge. 

```bash
pytest --junitxml=junit.xml -v qdscreen/tests/
ant -f ci_tools/generate-junit-html.xml
python ci_tools/generate-junit-badge.py
```

### PyPI Releasing memo

This project is now automatically deployed to PyPI when a tag is created. Anyway, for manual deployment we can use:

```bash
twine upload dist/* -r pypitest
twine upload dist/*
```
