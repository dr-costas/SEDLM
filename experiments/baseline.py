#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models import CRNN
from tools.printing import print_msg, print_date_and_time
from tools.various import get_argument_parser, CheckAllNone
from tools.file_io import load_settings_file

from ._processes import experiment

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['do_process']


@CheckAllNone()
def do_process(settings_path=None, settings=None):
    """The process of the baseline experiment.

    :param settings_path: The path for the settings.
    :type settings_path: str|None
    :param settings: The settings.
    :type settings: dict|None
    """
    if settings_path is not None:
        settings = load_settings_file(settings_path)

    print_msg('Starting baseline experiment', end='\n\n')
    experiment(settings, CRNN)


def main():
    print_date_and_time()

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    do_process(args.config_file)


if __name__ == '__main__':
    main()

# EOF
