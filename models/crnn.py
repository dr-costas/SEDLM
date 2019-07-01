#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import zeros
from torch.nn import Module, Sequential, GRUCell, \
    Linear, Dropout

from modules import dnn

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['CRNN']


class CRNN(Module):

    def __init__(self, cnn_channels, cnn_dropout, rnn_in_dim,
                 rnn_out_dim, rnn_dropout, nb_classes):
        """The CRNN model.

        :param cnn_channels: The amount of CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: The dropout to be applied to the CNNs.
        :type cnn_dropout: float
        :param rnn_in_dim: The input dimensionality of the RNN.
        :type rnn_in_dim: int
        :param rnn_out_dim: The output dimensionality of the RNN.
        :type rnn_out_dim: int
        :param rnn_dropout: The dropout to be applied to the RNN.
        :type rnn_dropout: float
        :param nb_classes: The amount of classes to be predicted.
        :type nb_classes: int
        """
        super(CRNN, self).__init__()

        self.dnn_output_features = cnn_channels
        self.rnn_hh_size = rnn_out_dim
        self.nb_classes = nb_classes

        self.dnn = dnn.DNN(cnn_channels=cnn_channels, cnn_dropout=cnn_dropout)
        self.rnn = Sequential(
            Dropout(rnn_dropout),
            GRUCell(rnn_in_dim, self.rnn_hh_size, bias=True)
        )
        self.classifier = Linear(self.rnn_hh_size, self.nb_classes, bias=True)

    def forward(self, x):
        """Forward pass of the CRNN model.

        :param x: The input to the CRNN.
        :type x: torch.Tensor
        :return: The output predictions.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = x.size()
        features = self.dnn(x).permute(0, 2, 1, 3).contiguous()
        features = features.view(b_size, t_steps, self.dnn_output_features)

        h = zeros(self.rnn_hh_size).to(x.device)
        outputs = zeros(b_size, t_steps, self.nb_classes).to(features.device)

        for t_step in range(t_steps):
            h = self.rnn(features[:, t_step, :], h)
            out = self.classifier(h)
            outputs[:, t_step, :] = out

        return outputs

# EOF
