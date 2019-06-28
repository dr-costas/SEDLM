#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path

import torch

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['save_model_state', 'load_model_state']


def save_model_state(base_path, model_f_name, model):
    """

    :param base_path:
    :type base_path: str
    :param model_f_name:
    :type model_f_name: str
    :param model:
    :type model: torch.nn.Module
    """
    torch.save(model.state_dict(), path.join(base_path, model_f_name))


def load_model_state(base_path, model_f_name, model):
    """

    :param base_path:
    :type base_path: str
    :param model_f_name:
    :type model_f_name: str
    :param model:
    :type model: torch.nn.Module
    :return:
    :rtype: torch.nn.Module
    """
    model.load_state_dict(torch.load(path.join(base_path, model_f_name)))

    return model

# EOF
