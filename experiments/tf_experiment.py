#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

from models.tf_crnn import TFCRNN
from tools.printing import print_msg, print_date_and_time
from tools.various import CheckAllNone, get_argument_parser
from tools.file_io import load_settings_file

from ._processes import experiment

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['do_process']


@CheckAllNone()
def do_process(settings_path=None, settings=None):
    """The process of the experiment for the proposed method.

    :param settings_path: The path for the settings.
    :type settings_path: str|None
    :param settings: The settings to be used.
    :type settings: dict|None
    """
    if settings_path is not None:
        settings = load_settings_file(settings_path)

    print_msg('Starting teacher forcing experiment', end='\n\n')
    p_print = partial(print_msg, decorate_prv='*', decorate_nxt='*', end='\n\n')

    for i in range(4):
        settings['data_loader'].update({'data_fold': i + 1})
        p_print('Fold {}'.format(i + 1))
        experiment(settings, TFCRNN)


def main():
    print_date_and_time()

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    do_process(args.config_file)


if __name__ == '__main__':
    main()

# EOF
