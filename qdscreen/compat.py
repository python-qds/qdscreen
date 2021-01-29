import sys

try:
    from sklearn.feature_selection import SelectorMixin
except ImportError:  # older sklearn
    from sklearn.feature_selection.base import SelectorMixin


from sklearn.base import BaseEstimator

try:
    BaseEstimator._get_tags
except AttributeError:
    import inspect

    _DEFAULT_TAGS = {
        'non_deterministic': False,
        'requires_positive_X': False,
        'requires_positive_y': False,
        'X_types': ['2darray'],
        'poor_score': False,
        'no_validation': False,
        'multioutput': False,
        "allow_nan": False,
        'stateless': False,
        'multilabel': False,
        '_skip_test': False,
        '_xfail_checks': False,
        'multioutput_only': False,
        'binary_only': False,
        'requires_fit': True,
        'requires_y': False,
    }

    def _more_tags(self):
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    # set the missing methods
    BaseEstimator._more_tags = _more_tags
    BaseEstimator._get_tags = _get_tags

try:
    BaseEstimator._validate_data
except AttributeError:
    from sklearn.utils.validation import check_X_y, check_array

    def _validate_data(self, X, y=None, reset=True,
                       validate_separately=False, **check_params):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """

        if y is None:
            # note: for some reason our patch above does not add this tag,
            # there is still a KeyError if we use self._get_tags()['requires_y']
            if self._get_tags().get('requires_y', False):
                raise ValueError(
                    "This {} estimator "
                    "requires y to be passed, but the target y is None."
                ).__format__(self.__class__.__name__)
            X = check_array(X, **check_params)
            out = X
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = check_array(X, **check_X_params)
                y = check_array(y, **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)

        return out

    # set the missing method
    BaseEstimator._validate_data = _validate_data


try:
    BaseEstimator._check_n_features
except AttributeError:
    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            Else, the attribute must already exist and the function checks
            that it is equal to `X.shape[1]`.
        """
        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
        else:
            if not hasattr(self, 'n_features_in_'):
                raise RuntimeError(
                    "The reset parameter is False but there is no "
                    "n_features_in_ attribute. Is this estimator fitted?"
                )
            if n_features != self.n_features_in_:
                raise ValueError(
                    'X has {} features, but this {} is expecting {} features '
                    'as input.'.format(n_features, self.__class__.__name__,
                                       self.n_features_in_)
                )

    # set the missing method
    BaseEstimator._check_n_features = _check_n_features


PY2 = sys.version_info < (3, 0)


def encode_if_py2(fun):
    """
    A decorator to use typically on __str__ and __repr__ methods if they return unicode literal string, so that
    under python 2 their result is encoded into utf-8 to avoir a
    UnicodeEncodeError: ... codec can't encode characters in position ...

    :param fun:
    :return:
    """
    if PY2:
        def new_fun(*args, **kwargs):
            return fun(*args, **kwargs).encode("utf-8")
        return new_fun
    else:
        return fun


def python_2_unicode_compatible(klass):
    """
    A decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.

    To support Python 2 and 3 with a single code base, define a __str__ method
    returning text and apply this decorator to the class.
    """
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied "
                             "to %s because it doesn't define __str__()." %
                             klass.__name__)
        klass.__unicode__ = klass.__str__
        def __str__(self):
            return self.__unicode__().encode('utf-8')

        klass.__str__ = __str__
    return klass
