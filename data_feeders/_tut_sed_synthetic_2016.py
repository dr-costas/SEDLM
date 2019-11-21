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
    def __init__(self, root_dir, split, input_features_file_name,
                 target_values_input_name, is_test):
        """TUT SED Synthetic 2016 dataset class. 
        
        :param root_dir: The root directory for the dataset. 
        :type root_dir: str
        :param split: The split for the dataset (e.g. training).
        :type split: str
        :param input_features_file_name: Input features directory name.
        :type input_features_file_name: str
        :param target_values_input_name: Target values directory name.
        :type target_values_input_name: str
        """
        super(TUTSEDSynthetic2016, self).__init__()
        data_path = Path(root_dir, 'synthetic', split)

        x_path = data_path.joinpath(input_features_file_name)
        y_path = data_path.joinpath(target_values_input_name)

        self.x = file_io.load_numpy_object(x_path)
        self.y = file_io.load_numpy_object(y_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

# EOF
