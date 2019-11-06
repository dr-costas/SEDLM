#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DNN']


class DNN(nn.Module):

    def __init__(self, cnn_channels, cnn_dropout):

        super(DNN, self).__init__()

        layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2
            ), nn.BatchNorm2d(cnn_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        )

        layer_2 = nn.Sequential(
            nn.Dropout2d(cnn_dropout),
            nn.Conv2d(
                in_channels=cnn_channels, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2
            ), nn.BatchNorm2d(cnn_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )

        layer_3 = nn.Sequential(
            nn.Dropout2d(cnn_dropout),
            nn.Conv2d(
                in_channels=cnn_channels, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2
            ), nn.BatchNorm2d(cnn_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout2d(cnn_dropout)
        )

        self.dnn = nn.Sequential(layer_1, layer_2, layer_3)

    def forward(self, x):
        """The forward pass of the DNN.

        :param x: The input audio features.
        :type x: torch.Tensor
        :return: The latent learned representation\
                 by the DNN.
        :rtype: torch.Tensor
        """
        return self.dnn(x.unsqueeze(1))

# EOF
