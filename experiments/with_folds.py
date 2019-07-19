#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

from models import CRNN, TFCRNN
from tools.printing import print_msg, print_date_and_time
from tools.various import CheckAllNone, get_argument_parser
from tools.file_io import load_settings_file

from ._processes import experiment

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['do_process']


@CheckAllNone()
def do_process(settings_path=None, settings=None, use_tf=False):
    """The process of the experiment for the proposed method.

    :param settings_path: The path for the settings.
    :type settings_path: str|None
    :param settings: The settings to be used.
    :type settings: dict|None
    :param use_tf: Do we use teacher forcing?
    :type use_tf: bool
    :param use_tf: Do we use teacher forcing?
    :type use_tf: bool
    """
    if settings_path is not None:
        settings = load_settings_file(settings_path)

    model = TFCRNN if use_tf else CRNN

    if not use_tf:
        print_msg('Baseline experiment')
    print_msg('Starting experiment with folds', end='\n\n')
    p_print = partial(print_msg, decorate_prv='*', decorate_nxt='*', end='\n\n')

    for i in range(2, 4):
        settings['data_loader'].update({'data_fold': i + 1})
        p_print('Fold {}'.format(i + 1))
        experiment(settings, model, use_tf=use_tf)


def main():
    print_date_and_time()

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    do_process(args.config_file, use_tf=not args.baseline)


if __name__ == '__main__':
    main()

# EOF
