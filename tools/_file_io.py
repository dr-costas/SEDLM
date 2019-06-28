#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import shutil
import pickle
import yaml
import numpy as np
from librosa import load

__author__ = 'Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = [
    'dump_pickle_file', 'load_pickle_file', 'read_txt_file',
    'check_and_create_dir', 'is_wav_file', 'get_all_wav_files',
    'load_audio_file', 'dump_numpy_object', 'change_parent_path',
    'load_numpy_object', 'get_all_numpy_files', 'is_of_that_extension',
    'delete_dir', 'load_yaml_file', 'load_settings_file'
]


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


def dump_pickle_file(obj, file_name, protocol=2):
    """Dumps an object to pickle file.

    :param obj: The object to dump.
    :type obj: object | list | dict | numpy.ndarray
    :param file_name: The resulting file name.
    :type file_name: str
    :param protocol: The protocol to be used.
    :type protocol: int
    """
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_txt_file(file_name):
    """Reads a text (.txt) file and returns the contents.

    :param file_name: The file name of the txt file.
    :type file_name: str
    :return: The contents of the file.
    :rtype: list[str]
    """
    with open(file_name) as f:
        return f.readlines()


def check_and_create_dir(the_dir):
    """Checks if a directory exists and creates it if not.

    :param the_dir: The path of the dir.
    :type the_dir: str
    """
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)


def is_wav_file(this_file):
    """Checks if the specified file is a .wav file./

    :param this_file: The path of the file to be checked.
    :type this_file: str
    :return: The indication if is or not a .wav file.
    :rtype: bool
    """
    return os.path.splitext(this_file)[-1] == '.wav'


def get_all_wav_files(from_this_dir):
    """Returns all the paths of the audio files \
    from the specified directory.

    :param from_this_dir: The directory.
    :type from_this_dir: str
    :return: The paths of the audio files.
    :rtype: list[str]
    """
    return sorted(os.path.join(from_this_dir, a_file)
                  for a_file in os.listdir(from_this_dir)
                  if is_wav_file(a_file))


def load_audio_file(audio_file, sr, mono, offset=0.0, duration=None):
    """Loads the data of an audio file.

    :param audio_file: The path of the audio file.
    :type audio_file: str
    :param sr: The sampling frequency to be used.
    :type sr: int
    :param mono: Turn to mono?
    :type mono: bool
    :param offset: Offset to be used (in seconds).
    :type offset: float
    :param duration: Duration of signal to load (in seconds).
    :type duration: float|None
    :return: The audio data.
    :rtype: numpy.ndarray
    """
    return load(
        path=audio_file, sr=sr, mono=mono, offset=offset,
        duration=duration
    )[0]


def dump_numpy_object(np_obj, file_name, ext='.npy', replace_ext=True):
    """Dumps a numpy object to HDD.

    :param np_obj: The numpy object.
    :type np_obj: numpy.ndarray
    :param file_name: The file name to be used.
    :type file_name: str
    :param ext: The extension for the dumped object.
    :type ext: str
    :param replace_ext: Replace extension if `file_name`\
                        has a different one?
    :type replace_ext: bool
    """
    np.save('{}{}'.format(os.path.splitext(file_name)[0], ext)
            if replace_ext and (os.path.splitext(file_name)[-1] != ext
                                or os.path.splitext(file_name)[-1] == '')
            else file_name, np_obj)


def load_numpy_object(f_name):
    """Loads anf returns a numpy object.

    :param f_name: The path of the object.
    :type f_name: str
    :return: The numpy object.
    :rtype: numpy.ndarray
    """
    return np.load(f_name)


def change_parent_path(a_path, new_parent):
    """Changes the parent of a path.

    :param a_path: The path
    :type a_path: str
    :param new_parent: The new parent.
    :type new_parent: str
    :return: The path with the new parent.
    :rtype: str
    """
    return os.path.join(
        new_parent,
        os.path.split(a_path)[-1]
    )


def is_of_that_extension(this_file, ext='.npy'):
    """Checks if the specified file is of a given extension.

    :param this_file: The path of the file to be checked.
    :type this_file: str
    :param ext: The extension for the file.
    :type ext: str
    :return: The indication if is or not a file of the given extension.
    :rtype: bool
    """
    return os.path.splitext(this_file)[-1] == ext


def get_all_numpy_files(from_this_dir, ext='.npy'):
    """Returns all the paths of the numpy files \
    from the specified directory.

    :param from_this_dir: The directory.
    :type from_this_dir: str
    :param ext: The extension for the numpy files.
    :type ext: str
    :return: The paths of the numpy files.
    :rtype: list[str]
    """
    return sorted(os.path.join(from_this_dir, a_file)
                  for a_file in os.listdir(from_this_dir)
                  if is_of_that_extension(a_file, ext))


def delete_dir(a_dir):
    """Deletes a directory.

    :param a_dir: The directory to delete.
    :type a_dir: str
    """
    shutil.rmtree(a_dir)


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
