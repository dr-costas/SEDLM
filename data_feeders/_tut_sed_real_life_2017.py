#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from torch.utils.data import Dataset
import numpy as np

from tools import file_io

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['TUTSEDRealLife2017']


class TUTSEDRealLife2017(Dataset):
    """TUT SED Real Life 2017.
    """
    def __init__(self, dataset_root_dir, dataset_split, data_fold,
                 normalized_features="True"):
        """TUT SED Real Life 2017 dataset class.

        :param dataset_root_dir: The dataset root directory.
        :type dataset_root_dir: pathlib.Path
        :param dataset_split: The split that we want.
        :type dataset_split: str
        :param data_fold: The data fold.
        :type data_fold: int
        :param normalized_features: Get the normalized features?
        :type normalized_features: bool
        """
        seq_len = 1024

        the_split = 'train' if dataset_split == 'training' else 'test'
        split_dir = 'dev'

        data_path = Path().joinpath(
            dataset_root_dir, 'real_life', '{}_{}'.format(
                split_dir, 2017), 'fold{}'.format(data_fold))

        path_ending = '{}{}_features'.format(
            the_split, '_normalized' if normalized_features else '')

        all_data_paths = sorted([a_path for a_path in data_path.iterdir()
                                 if a_path.stem.endswith(path_ending)])

        all_data = [file_io.load_pickle_file(str(i)) for i in all_data_paths]

        self.x = np.concatenate([f for i in all_data for f in i['features']])
        self.y = np.concatenate([l for i in all_data for l in i['labels']])

        nb_sequences, red = divmod(self.x.shape[0], seq_len)

        self.x = np.concatenate([
            self.x, np.zeros((seq_len - red, self.x.shape[-1]))
        ]).reshape((-1, seq_len, self.x.shape[-1]))
        self.y = np.concatenate([
            self.y, np.zeros((seq_len - red, self.y.shape[-1]))
        ]).reshape((-1, seq_len, self.y.shape[-1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

# EOF
