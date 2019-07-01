#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools.printing import print_date_and_time, print_yaml_settings
from tools.various import get_argument_parser
from tools.file_io import load_settings_file
from experiments.tf_experiment import do_process as tf_process
from experiments.baseline import do_process as baseline

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = []


def main():
    """Main entry point for the project.
    """
    print_date_and_time()

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    experiment_process = baseline if args.baseline else tf_process

    settings = load_settings_file(args.config_file)
    print_yaml_settings(settings)

    experiment_process(settings=settings)


if __name__ == '__main__':
    main()

# EOF
