#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader

from ._tut_sed_synthetic_2016 import TUTSEDSynthetic2016
from ._tut_sed_real_life_2017 import TUTSEDRealLife2017
from ._tut_sed_real_life_2016 import TUTSEDRealLife2016

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_tut_sed_data_loader']


def get_tut_sed_data_loader(dataset_root_dir, dataset_split,
                            batch_size, shuffle, drop_last,
                            normalized_features=True,
                            data_version='synthetic',
                            data_fold=None, scene=None):
    """Creates and returns the data loader.

    :param dataset_root_dir: The root dir for the dataset.
    :type dataset_root_dir: str
    :param dataset_split: The split of the data (training, \
                          validation, or testing).
    :type dataset_split: str
    :param batch_size: The batch size.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :param drop_last: Drop last examples?
    :type drop_last: bool
    :param normalized_features: Use SMUV features?
    :type normalized_features: bool
    :param data_version: Which version of the dataset? Accepted\
                         values are `synthetic` and `real_file`.
    :type data_version: str
    :param data_fold: Which fold?
    :type data_fold: int
    :param scene: Which scene?
    :type scene: str
    :return: The TUT BREACNNModel data loader.
    :rtype: torch.utils.data.DataLoader
    """
    common_kwargs = {
        'root_dir': dataset_root_dir,
        'split': dataset_split,
        'norm_features': normalized_features}

    if data_version == 'synthetic':
        dataset = TUTSEDSynthetic2016(**common_kwargs)
    else:
        if data_version == 2016:
            dataset = TUTSEDRealLife2016(
                data_fold=data_fold, scene=scene,
                **common_kwargs)
        else:
            dataset = TUTSEDRealLife2017(
                data_fold=data_fold, **common_kwargs)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle if dataset_split == 'training' else False,
        drop_last=drop_last)

# EOF
