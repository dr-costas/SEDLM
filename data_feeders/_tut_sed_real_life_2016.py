#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from torch.utils.data import Dataset
import numpy as np

from tools import file_io

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['TUTSEDRealLife2016']


class TUTSEDRealLife2016(Dataset):
    """TUT SED Real Life 2016 dataset
    """
    def __init__(self, root_dir, split, data_fold, scene,
                 norm_features="True"):
        """TUT SED Real Life 2016 dataset class.

        :param root_dir: The root directory for the dataset.
        :type root_dir: str
        :param split: The split for the dataset (e.g. training).
        :type split: str
        :param data_fold: The data fold.
        :type data_fold: int
        :param scene: The acoustic scene.
        :type scene: str
        :param norm_features: Get the normalized features?
        :type norm_features: bool
        """
        seq_len = 1024

        the_split = 'train' if split == 'training' else 'test'

        data_path = Path().joinpath(
            root_dir, 'real_life', '{}_{}'.format(
                'dev', 2016), 'fold{}'.format(data_fold))

        all_data_paths = sorted([
            a_path for a_path in data_path.iterdir()
            if a_path.stem.endswith('{}{}_features'.format(
                the_split, '_normalized' if norm_features else ''))
            and a_path.stem.startswith(scene)])

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
