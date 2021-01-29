# -*- coding: utf-8 -*-
# the above encoding declaration is needed to have non-ascii characters in this file (anywhere even in comments)
from __future__ import unicode_literals

# this module mimics what happens in main.py:
# - there are special characters in the Foo.__str__ method. To support them in python 2 we need the line 1 (coding)
# - since in python 2 __str__ is supposed to return bytes, we use @python_2_unicode_compatible
# - since the special characters in the Foo.__str__ are not ascii, we need them to be declared as unicode literal. For
#   this we can either do it by using a u as in 'u"└─"' or by importing unicode_literals from __future__ (line 2).
#   Note: if we want to have some strings as unicode and some others not we MUST use u"" and not this import.

from qdscreen.compat import python_2_unicode_compatible, encode_if_py2


# @python_2_unicode_compatible
class Foo(object):
    @encode_if_py2
    def __str__(self):
        return u"└─ab\n"

    # def toto(self):
    #     return "fjdlkdlms"

    def __repr__(self):
        return str(self)  # + self.toto()
