#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ._real_life_dataset import SEDRealLife

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['TUTSEDRealLife2016']


class TUTSEDRealLife2016(SEDRealLife):
    """TUT SED Real Life 2016 dataset
    """
    def __init__(self, root_dir, data_fold, scene,
                 input_features_file_name, target_values_input_name,
                 is_test):
        """TUT SED Real Life 2016 dataset class.

        :param root_dir: The root directory for the dataset.
        :type root_dir: str
        :param data_fold: The data fold.
        :type data_fold: int
        :param scene: The acoustic scene.
        :type scene: str
        :param input_features_file_name: Input features file name.
        :type input_features_file_name: str
        :param target_values_input_name: Target values file name.
        :type target_values_input_name: str
        :param is_test: Want the test split?
        :type is_test: bool
        """
        super(TUTSEDRealLife2016, self).__init__(
            root_dir=root_dir, data_dir='real_life_2016',
            data_fold=data_fold,
            scene=scene,
            input_features_file_name=input_features_file_name,
            target_values_input_name=target_values_input_name,
            seq_len=1024, is_test=is_test
        )

# EOF
