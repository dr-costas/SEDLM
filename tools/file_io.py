#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import pickle
import yaml
import numpy as np

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['load_pickle_file', 'load_numpy_object',
           'load_yaml_file', 'load_settings_file']


def load_pickle_file(file_name, encoding='latin1'):
    """Loads a pickle file.

    :param file_name: The file name (extension included).
    :type file_name: pathlib.Path|str
    :param encoding: The encoding of the file.
    :type encoding: str
    :return: The loaded object.
    :rtype: object | list | dict | numpy.ndarray
    """
    if type(file_name) == str:
        file_name = pathlib.Path(file_name)
    with file_name.open('rb') as f:
        return pickle.load(f, encoding=encoding)


def load_numpy_object(f_name):
    """Loads anf returns a numpy object.

    :param f_name: The path of the object.
    :type f_name: str
    :return: The numpy object.
    :rtype: numpy.ndarray
    """
    return np.load(f_name)


def load_yaml_file(file_path):
    """Reads and returns the contents of a YAML file.

    :param file_path: The path to the YAML file.
    :type file_path: pathlib.Path|str
    :return: The contents of the YAML file.
    :rtype: dict
    """
    if type(file_path) == str:
        file_path = pathlib.Path(file_path)

    with file_path.open('r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def load_settings_file(file_name, settings_dir=pathlib.Path('settings')):
    """Reads and returns the contents of a YAML settings file.

    :param file_name: The name of the settings file.
    :type file_name: str
    :param settings_dir: The directory with the settings files.
    :type settings_dir: pathlib.Path|str
    :return: The contents of the YAML settings file.
    :rtype: dict
    """
    settings_dir = pathlib.Path(settings_dir) \
        if type(settings_dir) == str else settings_dir
    settings_file_path = settings_dir.joinpath('{}.yaml'.format(file_name))
    return load_yaml_file(settings_file_path)

# EOF
