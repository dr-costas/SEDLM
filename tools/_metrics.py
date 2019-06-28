#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from sklearn.metrics import accuracy_score

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = [
    'accuracy',
    'f1_per_frame',
    'error_rate_per_second',
    'error_rate_per_frame',
    'f1',
    'f1_per_second',
    'precision',
    'recall',
    'tp_tf_fp_fn'
]


def accuracy(y_hat, y_true):
    """

    Calculates the accuracy from the `y_hat` predictions\
    and the `y_true` ground truth values.

    :param y_hat: The predictions
    :type y_hat: torch.Tensor
    :param y_true: The ground truth values
    :type y_true: torch.Tensor
    :return: The mean accuracy
    :rtype: float
    """
    y_hat_ready = y_hat.max(-1)[1] if y_hat.ndimension() > 1 else y_hat.ge(0.5).long()

    if y_true.device.type != 'cpu':
        y_true = y_true.cpu()
        y_hat_ready = y_hat_ready.cpu()

    y = y_true.numpy()
    y_hat_ready = y_hat_ready.numpy()

    the_accuracy = accuracy_score(y, y_hat_ready)

    return the_accuracy


def f1(tp, fp, fn):
    """

    Gets the F1 score from the TP, FP, and FN.

    :param tp: The TP
    :type tp: torch.Tensor
    :param fp: The FP
    :type fp: torch.Tensor
    :param fn: The FN
    :type fn: torch.Tensor
    :return: The F1 score
    :rtype: torch.Tensor
    """
    if all([m.sum().item() == 0 for m in [tp, fp, fn]]):
        return 0
    f1_nominator = tp.mul(2)
    f1_denominator = tp.mul(2).add(fn).add(fp)
    return f1_nominator.div(f1_denominator + torch.finfo(torch.float32).eps)


def f1_per_second(y_hat, y_true, frames=86):
    """

    Gets the F1 score per second, based on\
    TP, FP, and FN, calculated from the `y_hat`\
    predictions and `y_true` ground truth values.

    :param y_hat: The predictions
    :type y_hat: torch.Tensor
    :param y_true: The ground truth values
    :type y_true: torch.Tensor
    :param frames: Amount of frames in one second
    :type frames: int
    :return: The F1 score per second
    :rtype: torch.Tensor
    """
    t_thr = torch.floor(torch.ones(1) * y_true.size()[1]/frames).int()
    f_size = y_true.size()[-1]
    y_hat_second = y_hat[:, :t_thr * frames, :].contiguous().view(-1, t_thr, f_size)
    y_true_second = y_true[:, :t_thr * frames, :].contiguous().view(-1, t_thr, f_size)

    return f1_per_frame(y_hat_second, y_true_second)


def f1_per_frame(y_hat, y_true):
    """

    Gets the average per frame F1 score, based on\
    TP, FP, and FN, calculated from the `y_hat`\
    predictions and `y_true` ground truth values.

    :param y_hat: The predictions
    :type y_hat: torch.Tensor
    :param y_true: The ground truth values
    :type y_true: torch.Tensor
    :return: The F1 score per frame
    :rtype: torch.Tensor
    """
    tp, _, fp, fn = tp_tf_fp_fn(y_hat=y_hat, y_true=y_true, dim_sum=None)
    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    the_f1 = f1(tp=tp, fp=fp, fn=fn)
    return the_f1


def precision(tp, fp):
    """

    Calculates the precision from TP and FP

    :param tp: The true positives
    :type tp: torch.Tensor
    :param fp: The false positives
    :type fp: torch.Tensor
    :return: The precision
    :rtype: torch.Tensor
    """
    if all([m.sum().item() == 0 for m in [tp, fp]]):
        return 0
    denominator = tp.add(fp)
    return tp.div(denominator + torch.finfo(torch.float32).eps)


def recall(tp, fn):
    """

    Calculates the recall from TP and FN

    :param tp: The true positives
    :type tp: torch.Tensor
    :param fn: The false negatives
    :type fn: torch.Tensor
    :return: The recall
    :rtype: torch.Tensor
    """
    if all([m.sum().item() == 0 for m in [tp, fn]]):
        return 0
    denominator = tp.add(fn)
    return tp.div(denominator + torch.finfo(torch.float32).eps)


def tp_tf_fp_fn(y_hat, y_true, dim_sum):
    """

    Gets the true positive (TP), true negative (TN),\
    false positive (FP), and false negative (FN).

    :param y_hat: The predictions
    :type y_hat: torch.Tensor
    :param y_true: The ground truth values
    :type y_true: torch.Tensor
    :param dim_sum: Dimension to sum TP, TN, FP, and FN. If\
                    it is None, then the default behaviour from\
                    PyTorch`s sum is assumed.
    :type dim_sum: int | None
    :return: The TP, TN, FP, FN.
    :rtype: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    y_hat_positive = y_hat.ge(0.5)
    y_hat_negative = y_hat.lt(0.5)

    y_true_positive = y_true.eq(1.)
    y_true_negative = y_true.eq(0.)

    tp = y_hat_positive.mul(y_true_positive).view(-1, y_hat_positive.size()[-1]).float()
    tn = y_hat_negative.mul(y_true_negative).view(-1, y_hat_positive.size()[-1]).float()

    fp = y_hat_positive.mul(y_true_negative).view(-1, y_hat_positive.size()[-1]).float()
    fn = y_hat_negative.mul(y_true_positive).view(-1, y_hat_positive.size()[-1]).float()

    if dim_sum is not None:
        tp = tp.sum(dim=dim_sum)
        tn = tn.sum(dim=dim_sum)
        fp = fp.sum(dim=dim_sum)
        fn = fn.sum(dim=dim_sum)

    return tp, tn, fp, fn


def error_rate_per_second(y_hat, y_true, frames=86):
    """Calculates the error rate based on FN and FP,
    for one second.

    :param y_hat: The predictions.
    :type y_hat: torch.Tensor
    :param y_true: The ground truth.
    :type y_true: torch.Tensor
    :param frames: The amount of frames in one second.
    :type frames: int
    :return: The error rate.
    :rtype: float
    """
    t_thr = torch.floor(torch.ones(1) * y_true.size()[1]/frames).int()
    f_size = y_true.size()[-1]
    y_hat_second = y_hat[:, :t_thr * frames, :].contiguous().view(-1, t_thr, f_size)
    y_true_second = y_true[:, :t_thr * frames, :].contiguous().view(-1, t_thr, f_size)

    return error_rate_per_frame(y_hat_second, y_true_second)


def error_rate_per_frame(y_hat, y_true):
    """Calculates the error rate based on FN and FP,
    for one second.

    :param y_hat: The predictions.
    :type y_hat: torch.Tensor
    :param y_true: The ground truth.
    :type y_true: torch.Tensor
    :param frames: The amount of frames in one second.
    :type frames: int
    :return: The error rate.
    :rtype: float
    """
    _, __, fp, fn = tp_tf_fp_fn(y_hat, y_true, -1)

    s = fn.min(fp).sum()
    d = fn.sub(fp).clamp_min(0).sum()
    i = fp.sub(fn).clamp_min(0).sum()
    n = y_true.sum() + torch.finfo(torch.float32).eps

    return (s + d + i)/n


def main():
    import numpy as np
    from sklearn.metrics import f1_score

    print('Comparing 2D matrices with scikit learn F1 micro')
    print('=' * 50)
    for i in range(10):
        y_true = np.random.randint(0, 2, (1024, 16))
        y_hat = np.random.randint(0, 2, (1024, 16))

        y_true_v = torch.autograd.Variable(torch.from_numpy(y_true).float())
        y_hat_v = torch.autograd.Variable(torch.from_numpy(y_hat).float())

        # f1_o = get_overall_frame_based_f1(y_hat_v, y_true_v).data[0]
        f1_f = f1_per_frame(y_hat_v, y_true_v).data[0]
        f1_micro = f1_score(y_true, y_hat, average='micro')

        print('Diff F1_f - F1_micro: {:.8f}'.format(
                # np.abs(f1_o - f1_micro),
                np.abs(f1_f - f1_micro)
              ))

    print(' ')
    print('Comparing 3D matrices with the two different methods')
    print('=' * 50)
    for i in range(10):
        y_true = np.random.randint(0, 2, (32, 1024, 16))
        y_hat = np.random.randint(0, 2, (32, 1024, 16))

        y_true_v = torch.autograd.Variable(torch.from_numpy(y_true).float())
        y_hat_v = torch.autograd.Variable(torch.from_numpy(y_hat).float())

        # # f1_o = get_overall_frame_based_f1(y_hat_v, y_true_v).data[0]
        # f1_f = f1_per_frame(y_hat_v, y_true_v).data[0]
        #
        # print('Diff F1_f - F1_o: {:.8f}'.format(
        #         np.abs(f1_f - f1_o),
        #       ))


if __name__ == '__main__':
    main()

# EOF
