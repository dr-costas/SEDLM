#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader

from ._tut_sed_synthetic_2016 import TUTSEDSynthetic2016
from ._tut_sed_real_life_2017 import TUTSEDRealLife2017
from ._tut_sed_real_life_2016 import TUTSEDRealLife2016

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_tut_sed_data_loader']


def get_tut_sed_data_loader(root_dir, split, data_version, batch_size,
                            shuffle, drop_last, input_features_file_name,
                            target_values_input_name, data_fold=None,
                            scene=None, is_test=False):
    """Creates and returns the data loader.

    :param root_dir: The root dir for the dataset.
    :type root_dir: str
    :param split: The split of the data (training, \
                          validation, or testing).
    :type split: str
    :param data_version: Which version of the dataset? Accepted\
                         values are `synthetic` and `real_file`.
    :type data_version: str
    :param batch_size: The batch size.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :param drop_last: Drop last examples?
    :type drop_last: bool
    :param input_features_file_name: Input features file name.
    :type input_features_file_name: str
    :param target_values_input_name: Target values file name.
    :type target_values_input_name: str
    :param data_fold: Which fold?
    :type data_fold: int
    :param scene: Which scene?
    :type scene: str
    :param is_test: We want the testing split for folds case?
    :type is_test: bool
    :return: The TUT BREACNNModel data loader.
    :rtype: torch.utils.data.DataLoader
    """
    common_kwargs = {
        'root_dir': root_dir,
        'input_features_file_name': input_features_file_name,
        'target_values_input_name': target_values_input_name
    }

    if data_version == 'synthetic':
        common_kwargs.update({'split': split})
        dataset = TUTSEDSynthetic2016(**common_kwargs)
    else:
        common_kwargs.update({
            'data_fold': data_fold,
            'is_test': is_test
        })

        if data_version == 2016:
            common_kwargs.update({'scene': scene})
            dataset = TUTSEDRealLife2016(**common_kwargs)
        else:
            dataset = TUTSEDRealLife2017(**common_kwargs)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle if split == 'training' else False,
        drop_last=drop_last)

# EOF
