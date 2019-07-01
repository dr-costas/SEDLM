#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from torch.utils.data import Dataset

from tools import file_io

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['TUTSEDSynthetic2016']


class TUTSEDSynthetic2016(Dataset):
    """TUT SED Synthetic 2016 dataset
    """
    def __init__(self, root_dir, split, norm_features="True"):
        """TUT SED Synthetic 2016 dataset class. 
        
        :param root_dir: The root directory for the dataset. 
        :type root_dir: str
        :param split: The split for the dataset (e.g. training).
        :type split: str
        :param norm_features: Get the normalized features?
        :type norm_features: bool
        """
        if split not in ['training', 'validation', 'testing']:
            raise AssertionError('Split `{}` not recognized. Use one of: {}'.format(
                split, ', '.join(['training', 'validation', 'testing'])))

        data_path = Path().joinpath(root_dir, 'synthetic', split)
        x_path = data_path.joinpath('features{norm}.npy'.format(
            norm='_normalized' if norm_features else ''))

        y_path = data_path.joinpath('target_values.npy')

        self.x = file_io.load_numpy_object(x_path)
        self.y = file_io.load_numpy_object(y_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

# EOF
