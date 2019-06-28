#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['get_argument_parser']


def get_argument_parser():
    """Creates and returns the ArgumentParser for this project.

    :return: The argument parser.
    :rtype: argparse.ArgumentParser
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config-file', type=str, default='')

    return arg_parser

# EOF
