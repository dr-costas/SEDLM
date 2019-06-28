#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import librosa

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['stft', 'mel_band_energies']


def stft(signal, n_fft, hop, win_func, center=False):
    """Calculates the STFT for a given signal.

    :param signal: The signal to use.
    :type signal: numpy.ndarray
    :param n_fft: The FFT points.
    :type n_fft: int
    :param hop: The hop in samples.
    :type hop: int
    :param win_func: The windowing function to be used.
    :type win_func: str
    :param center: Center windowing frames?
    :type center: bool
    :return: The STFT of the given signal.
    :rtype: numpy.ndarray
    """
    return librosa.core.stft(
        y=signal, n_fft=n_fft, hop_length=hop,
        window=win_func, center=center
    )


def mel_band_energies(signal, nb_filters, n_fft, sr, hop, win_fun, center=False, htk=False):
    """Calculates the MEL band energies for a given signal.

    :param signal: The signal to be used.
    :type signal: numpy.ndarray
    :param nb_filters: The amount of MEL filters to be used.
    :type nb_filters: int
    :param n_fft: The FFT points.
    :type n_fft: int
    :param sr: The sampling frequency used for the signal in Hz.
    :type sr: int
    :param hop: The hop size in samples.
    :type hop: int
    :param win_fun: The windowing function to be used.
    :type win_fun: str
    :param center: Center windowing frames?
    :type center: bool
    :param htk: Use HTK formula?
    :type htk: bool
    :return: The MEL band energies for the given signal.
    :rtype: numpy.ndarray
    """
    spectrogram = np.abs(stft(signal, n_fft, hop, win_fun, center))
    return librosa.feature.melspectrogram(
        sr=sr, S=spectrogram, n_fft=n_fft, power=1.,
        n_mels=nb_filters, htk=htk
    )

# EOF
