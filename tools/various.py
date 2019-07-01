#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['CheckAllNone']


class CheckAllNone(object):
    """Decorator to assure that at least one argument is\
    not None.
    """
    def __init__(self):
        super(CheckAllNone, self).__init__()
        self.fn = None

    def _decorated(self, *args, **kwargs):
        all_args = list(*args)
        all_args.extend(kwargs.values())

        if all(an_arg is None for an_arg in all_args):
            raise AssertionError(
                'Provide at least one not None argument.')
        return self.fn(*args, **kwargs)

    def __call__(self, fn):
        self.fn = fn
        return self._decorated

# EOF
