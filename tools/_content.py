#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_file_index']


def get_file_index(file_name):
    """Return the file index of the file.

    :param file_name: The file name.
    :type file_name: str
    :return: The file index.
    :rtype: int
    """
    return int(file_name.split('mix_')[-1].split('_no')[0])

# EOF
