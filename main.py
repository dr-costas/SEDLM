#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools.printing import print_date_and_time, print_yaml_settings
from tools.various import get_argument_parser
from tools.file_io import load_settings_file
from experiments.with_folds import do_process as with_folds_process
from experiments.no_folds import do_process as no_folds_process

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = []


def main():
    """Main entry point for the project.
    """
    print_date_and_time()

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    settings = load_settings_file(args.config_file)
    print_yaml_settings(settings)

    experiment_process = with_folds_process if settings['global']['has_folds'] \
        else no_folds_process

    experiment_process(settings=settings, use_tf=not args.baseline)


if __name__ == '__main__':
    main()

# EOF
