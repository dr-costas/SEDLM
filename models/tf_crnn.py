#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module, GRUCell, Linear, Dropout
from torch import zeros, Tensor, cat

from ._modules import dnn

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['TFCRNN']


class TFCRNN(Module):

    def __init__(self, cnn_channels, cnn_dropout,
                 rnn_in_dim, rnn_out_dim, rnn_dropout,
                 nb_classes, batch_counter, gamma_factor,
                 mul_factor, min_prob, max_prob):
        """The Sound Event Detection (SED) model with teacher forcing and\
        scheduled sampling.

        :param cnn_channels: The amount of CNN channels for the SED model.
        :type cnn_channels: int
        :param cnn_dropout: The dropout percentage for the CNNs dropout.
        :type cnn_dropout: float
        :param rnn_in_dim: The input dimensionality for the RNNs.
        :type rnn_in_dim: int
        :param rnn_out_dim: The output dimensionality for the RNNs.
        :type rnn_out_dim: int
        :param rnn_dropout: The dropout percentage for the RNN dropout.
        :type rnn_dropout: float
        :param nb_classes: The amount of output classes.
        :type nb_classes: int
        :param batch_counter: The amount of batches in one full epoch.
        :type batch_counter: int
        :param gamma_factor: The gamma factor for scheduled sampling.
        :type gamma_factor: float
        :param mul_factor: The multiplication factor for scheduled sampling.
        :type mul_factor: float
        :param min_prob: The minimum probability for selecting predictions.
        :type min_prob: float
        :param max_prob: The maximum probability for selecting predictions.
        :type max_prob: float
        """
        super(TFCRNN, self).__init__()

        self.dnn_output_features = cnn_channels
        self.rnn_hh_size = rnn_out_dim
        self.nb_classes = nb_classes

        self.batch_counter = batch_counter
        self.gamma_factor = gamma_factor/mul_factor
        self._min_prob = 1 - min_prob
        self.max_prob = max_prob
        self.iteration = 0

        self.dnn = dnn.DNN(cnn_channels=cnn_channels, cnn_dropout=cnn_dropout)
        self.rnn_dropout = Dropout(rnn_dropout)
        self.rnn = GRUCell(rnn_in_dim + self.nb_classes, self.rnn_hh_size, bias=True)
        self.classifier = Linear(self.rnn_hh_size, self.nb_classes, bias=True)

    @property
    def min_prob(self):
        """Getter for the min_prob attribute.

        :return: The minimum probability for\
                 selecting predictions.
        :rtype: float
        """
        return 1 - self._min_prob

    @min_prob.setter
    def min_prob(self, value):
        """Setter for the min_prob attribute.

        :param value: The new value of the min_prob.
        :type value: float
        """
        self._min_prob = 1 - value

    def forward(self, x, y):
        """The forward pass of the CRNN model with\
        teacher forcing.

        :param x: The input audio features.
        :type x: torch.Tensor
        :param y: The predictions for teacher forcing.
        :type y: torch.Tensor
        :return: The predictions of TF CRNN.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = x.size()
        features = self.rnn_dropout(
            self.dnn(x).permute(0, 2, 1, 3).contiguous().view(
                b_size, t_steps, self.dnn_output_features))

        device = features.device

        h = zeros(self.rnn_hh_size).to(device)
        tf = zeros(b_size, self.output_size).to(device)
        flags = zeros(b_size).to(device)

        outputs = zeros(b_size, t_steps, self.nb_classes).to(device)

        for t_step in range(t_steps):

            prob = self.scheduled_sampling()
            flags.random_(0, 1001).div_(1000).lt_(prob)
            tf_input = cat([features[:, t_step, :], tf], dim=-1)

            h = self.rnn(tf_input, h)

            cls_out = self.classifier(h)
            sig_out = cls_out.sigmoid().gt(.5).float()

            try:
                for ii, flag in enumerate(flags):
                    tf[ii, :] = y[ii, t_step, :] if flag else sig_out[ii, :]
                self.iteration += 1
            except TypeError:
                tf[:, :] = sig_out

            outputs[:, t_step, :] = cls_out

        return outputs

    def scheduled_sampling(self):
        """Returns the probability to select
        the predicted value.

        :return: The current probability of\
                 selecting predictions.
        :rtype: torch.Tensor
        """
        p = self.iteration/self.batch_counter
        d = Tensor([-self.gamma_factor * p]).exp().item()
        return min(self.max_prob, 1 - min(self.min_prob, (2 / (1 + d)) - 1))

# EOF
