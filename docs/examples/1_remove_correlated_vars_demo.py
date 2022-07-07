#  Authors: Sylvain MARIE <sylvain.marie@se.com>, Thibaud RAHIER <t.rahier@criteo.com>
#            + All contributors to <https://github.com/python-qds/qdscreen/>
#
#  License: 3-clause BSD, <https://github.com/python-qds/qdscreen/blob/master/LICENSE>
"""
Removing correlated variables
=============================

In this example we show how to remove correlated categorical variables.

1. Strict determinism
---------------------

Let's consider the following dataset:
"""

import pandas as pd

df = pd.DataFrame({
   'U': ["a", "b", "d", "a", "b", "c", "a", "b", "d", "c"],
   'V': ["a", "b", "c", "a", "b", "c", "a", "b", "c", "c"],
   'W': ["a", "b", "b", "a", "b", "b", "a", "b", "b", "b"],
   'X': ["a", "a", "b", "b", "a", "c", "c", "a", "b", "c"],
   'Y': ["b", "b", "c", "c", "b", "a", "a", "b", "c", "a"],
   'Z': ["a", "a", "b", "a", "a", "b", "b", "a", "a", "b"]
})
print("Columns in df: %s" % list(df.columns))
df

# %%
# We can detect correlated categorical variables (functional dependencies):

from qdscreen import qd_screen

# detect strict deterministic relationships
qd_forest = qd_screen(df)
print(qd_forest)

# %%
# So with only features `U`, and `X` we should be able to predict `V`, `W`, and `Y`. `Z` is a root but has no children
# so it does not help.
#
# We can create a feature selection model from this deterministic forest object:

feat_selector = qd_forest.fit_selector_model(df)
feat_selector

# %%
# This model can be used to preprocess the dataset before a learning task:

only_important_features_df = feat_selector.remove_qd(df)
only_important_features_df

# %%
# It can also be used to restore the dependent columns from the remaining ones:

restored_full_df = feat_selector.predict_qd(only_important_features_df)
restored_full_df

# %%
# Note that the order of columns differs from origin, but apart from this,
# the restored dataframe is the same as the original:

pd.testing.assert_frame_equal(df, restored_full_df[df.columns])

# %%
# 2. Quasi determinism
# ---------------------
#
# In the above example, we used the default settings for `qd_screen`. By default only deterministic relationships are
# detected, which means that only variables that can perfectly be predicted (without loss of information) from others
# in the dataset are removed.
#
# In real-world datasets, some noise can occur in the data, or some very rare cases might happen, that you may wish to
# discard. Let's first look at the strength of the various relationships thanks to `keep_stats=True`:

# same than above, but this time remember the various indicators
qd_forest = qd_screen(df, keep_stats=True)

# display them
print(qd_forest.stats)

# %%
# In the last row of the last table (relative conditional entropies) we see that variable `Z`'s entropies decreases
# drastically to reach 28% of its initial entropy, if `X` or `Y` is known. So if we use quasi-determinism with relative
# threshold of 29% `Z` would be eliminated.

# detect quasi deterministic relationships
qd_forest2 = qd_screen(df, relative_eps=0.29)
print(qd_forest2)

# %%
# This time `Z` is correctly determined as being predictible from `X`.
#
# !!! note "equivalent nodes"
#     `X` and `Y` are equivalent variables so each of them could be the parent of the other. To avoid cycles so that the
#     result is still a forest (a set of trees), `X` was arbitrary selected as being the "representative" parent of all
#     its equivalents, and `Z` is attached to this representative parent.
#
# Another, easier way to detect that setting a relative threshold to 29% would eliminate `Z` is to print the
# conditional entropies in increasing order:

ce_df = qd_forest.get_entropies_table(from_to=False, sort_by="rel_cond_entropy")
ce_df.head(10)

# %%
# Or to use the helper plot function:

qd_forest.plot_increasing_entropies()

# %%
# 3. Integrating with scikit-learn
# --------------------------------
#
# `scikit-learn` is one of the most popular machine learning frameworks in python. It comes with a concept of
# `Pipeline` allowing you to chain several operators to make a model. `qdscreen` provides a `QDScreen` class for easy
# integration. It works exactly like other feature selection models in scikit-learn (e.g.
# [`VarianceThreshold`](https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold)):

from qdscreen.sklearn import QDScreen

X = [[0, 2, 0, 3],
     [0, 1, 4, 3],
     [0, 1, 1, 3]]

selector = QDScreen()
Xsel = selector.fit_transform(X)
Xsel

# %%

selector.inverse_transform(Xsel)
