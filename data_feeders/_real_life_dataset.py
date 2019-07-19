#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from torch.utils.data import Dataset
import numpy as np

from tools import file_io

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['SEDRealLife']


class SEDRealLife(Dataset):
    """Base class for real life datasets.
    """
    def __init__(self, root_dir, data_dir, data_fold, scene,
                 input_features_file_name, target_values_input_name,
                 seq_len, is_test):
        """Base class for real life datasets.

        :param root_dir: The root directory for the dataset.
        :type root_dir: str
        :param data_dir: The data directory for the dataset.
        :type data_dir: str
        :param data_fold: The data fold.
        :type data_fold: int
        :param scene: The acoustic scene (if applicable, else '').
        :type scene: str
        :param input_features_file_name: Input features file name.
        :type input_features_file_name: str
        :param target_values_input_name: Target values file name.
        :type target_values_input_name: str
        :param seq_len: Amount of feature vectors in one sequence.
        :type seq_len: int
        :param is_test: Want the test split?
        :type is_test: bool
        """
        super(SEDRealLife, self).__init__()

        data_path = Path(root_dir, data_dir, scene,
                         'fold_{}'.format(data_fold))

        f_prefix = 'test' if is_test else 'train'

        x_path = data_path.joinpath('{}_{}'.format(f_prefix, input_features_file_name))
        y_path = data_path.joinpath('{}_{}'.format(f_prefix, target_values_input_name))

        self.x = file_io.load_pickle_file(x_path)
        self.y = file_io.load_pickle_file(y_path)

        for i in range(len(self.x)):
            _, red = divmod(self.x[i].shape[0], seq_len)

            self.x[i] = np.concatenate([
                np.zeros((seq_len - red, self.x[i].shape[-1])), self.x[i]
            ]).reshape((-1, seq_len, self.x[i].shape[-1]))
            self.y[i] = np.concatenate([
                np.zeros((seq_len - red, self.y[i].shape[-1])), self.y[i]
            ]).reshape((-1, seq_len, self.y[i].shape[-1]))

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)

    def __len__(self):
        """The amount of examples in the dataset.

        :return: The amount of examples.
        :rtype: int
        """
        return len(self.x)

    def __getitem__(self, item):
        """Gets an example and its target values\
        from the dataset.

        :param item: Index of the example.
        :type item: int
        :return: The example and the target values.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        return self.x[item], self.y[item]

# EOF
