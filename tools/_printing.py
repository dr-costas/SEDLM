#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from contextlib import ContextDecorator
from datetime import datetime
from collections import namedtuple, Iterable

__author__ = 'Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = [
    'print_msg', 'inform_about_device', 'print_date_and_time',
    'print_processes_message', 'InformAboutProcess', 'print_yaml_settings',
    'print_training_results', 'print_evaluation_results', 'print_intro_messages',
    'print_empty_lines', 'print_modules_info'
]


_time_f_spec = '7.2'
_acc_f_spec = '6.2'
_loss_f_spec = '7.3'
_epoch_f_spec = '4'


def print_intro_messages(device):
    """Prints initial messages.

    :param device: The device to be used.
    :type device: str
    """
    print_date_and_time()
    print_msg(' ', start='')
    inform_about_device(device)
    print_msg(' ', start='')


def print_msg(the_msg, start='-- ', end='\n', flush=True,
              decorate_prv=None, decorate_nxt=None):
    """Prints a message.

    :param the_msg: The message.
    :type the_msg: str
    :param start: Starting decoration.
    :type start: str
    :param end: Ending character.
    :type end: str
    :param flush: Flush buffer now?
    :type flush: bool
    :param decorate_prv: Decoration for previous line.\
                         If argument is tuple or strings,\
                         then its element in the tuple is\
                         printed on one previous line.
    :type decorate_prv: tuple(str)|str|None
    :param decorate_nxt: Decoration for next line.\
                         If argument is tuple or strings,\
                         then its element in the tuple is\
                         printed on one previous line.
    :type decorate_nxt: tuple(str)|str|None
    """
    msg_len = len(str(the_msg))

    if decorate_prv is not None:
        dec = [decorate_prv] if type(decorate_prv) == str else decorate_prv
        for d in dec:
            print('{}{}'.format(start, d * msg_len), end='\n', flush=flush)

    msg_end = end if decorate_nxt is None else '\n'
    print('{}{}'.format(start, the_msg), end=msg_end, flush=flush)

    if decorate_nxt is not None:
        dec = [decorate_nxt] if type(decorate_nxt) == str else decorate_nxt
        for d in dec[:-1]:
            print('{}{}'.format(start, d * msg_len), end='\n', flush=flush)
        print('{}{}'.format(start, dec[-1] * msg_len), end=end, flush=flush)


def print_yaml_settings(the_settings):
    """Prints the settings in the YAML settings file.

    :param the_settings: The settings dict
    :type the_settings: dict
    """
    def _print_dict_yaml_settings(the_dict, indentation, start):
        """Prints a nested dict.

        :param the_dict: The nested dict.
        :type the_dict: dict
        :param indentation: Indentation for the printing.
        :type indentation: str
        :param start: Starting decoration.
        :type start: str
        """
        k_l_ = max(*([len(_k) for _k in the_dict.keys()] + [0]))
        for k_, v_ in the_dict.items():
            print_msg('{k_:<{k_l_:d}s}:'.format(k_=k_, k_l_=k_l_),
                      start='{}{}|-- '.format(start, indentation),
                      end=' ', flush=True)
            start = ''
            if type(v_) == dict:
                _print_dict_yaml_settings(v_, '{}{}'.format(indentation, ' ' * 5), '\n')
            elif type(v_) == list:
                print_msg(', '.join(map(str, v_)), start='')
            else:
                print_msg(v_, start='')

    try:
        print_msg('ModelDCASE description: {}'.format(the_settings['model_description']),
                  start='\n-- ')
    except KeyError:
        pass

    print_msg('Settings: ', end='\n\n')

    dict_to_print = {k__: v__ for k__, v__ in the_settings.items() if k__ != 'model_description'}
    k_len = max(*[len(k__) for k__ in dict_to_print.keys()])

    for k, v in dict_to_print.items():
        k_len = max(k_len, len(k))
        print_msg('{}:'.format(k), start=' ' * 2, end=' ')
        if type(v) == dict:
            _print_dict_yaml_settings(v, ' ' * 3, '\n')
        else:
            print_msg(v, start='')
        print_msg('', start='')


def inform_about_device(the_device):
    """Prints an informative message about the device that we are using.

    :param the_device: The device.
    :type the_device: str
    """
    from torch.cuda import get_device_name, current_device
    from platform import processor
    actual_device = get_device_name(current_device()) \
        if the_device.startswith('cuda') else processor()
    print_msg('Using device: `{}`.'.format(actual_device))


def print_date_and_time():
    """Prints the date and time of `now`.
    """
    print_msg(datetime.now().strftime('%Y-%m-%d %H:%M'), start='\n\n-- ', end='\n\n')


class InformAboutProcess(ContextDecorator):
    def __init__(self, starting_msg, ending_msg='done', start='-- ', end='\n'):
        """Context manager and decorator for informing about a process.

        :param starting_msg: The starting message, printed before the process starts.
        :type starting_msg: str
        :param ending_msg: The ending message, printed after process ends.
        :type ending_msg: str
        :param start: Starting decorator for the string to be printed.
        :type start: str
        :param end: Ending decorator for the string to be printed.
        :type end: str
        """
        super(InformAboutProcess, self).__init__()
        self.starting_msg = starting_msg
        self.ending_msg = ending_msg
        self.start_dec = start
        self.end_dec = end

    def __enter__(self):
        print_msg('{}... '.format(self.starting_msg), start=self.start_dec, end='')

    def __exit__(self, *exc_type):
        print_msg('{}.'.format(self.ending_msg), start='', end=self.end_dec)
        return False


def print_empty_lines(nb_lines=1):
    """Prints empty lines.

    :param nb_lines: The amount of lines.
    :type nb_lines: int
    """
    for _ in range(nb_lines):
        print_msg('', start='', end='\n')


def print_processes_message(workers):
    """Prints a message for how many processes are used.

    :param workers: The amount of processes.
    :type workers: int
    """
    msg_str = '| Using {} {} |'.format('single' if workers is None else workers,
                                       'processes' if workers > 1 else 'process')
    print_msg('\n'.join(['', '*' * len(msg_str), msg_str, '*' * len(msg_str)]), flush=True)


def print_training_results(epoch, training_loss, validation_loss,
                           training_f1, training_er,
                           validation_f1, validation_er,
                           time_elapsed):
    """Prints the results of the pre-training step to console.

    :param epoch: The epoch.
    :type epoch: int
    :param training_loss: The loss of the training data.
    :type training_loss: float
    :param validation_loss: The loss of the validation data.
    :type validation_loss: float | None
    :param training_f1: The F1 score for the training data.
    :type training_f1: float
    :param training_er: The error rate for the training data.
    :type training_er: float
    :param validation_f1: The F1 score for the validation data.
    :type validation_f1: float | None
    :param validation_er: The error rate for the validation data.
    :type validation_er: float | None
    :param time_elapsed: The time elapsed for the epoch.
    :type time_elapsed: float
    """
    the_msg = \
        'Epoch:{e:{e_spec}d} | ' \
        'Loss (tr/va):{l_tr:{l_f_spec}f}/{l_va:{l_f_spec}f} | ' \
        'F1 (tr/va):{f1_tr:{acc_f_spec}f}/{f1_va:{acc_f_spec}f} | ' \
        'ER (tr/va):{er_tr:{acc_f_spec}f}/{er_va:{acc_f_spec}f} | ' \
        'Time:{t:{t_f_spec}f} sec.'.format(
            e=epoch,
            l_tr=training_loss, l_va='None' if validation_loss is None else validation_loss,
            f1_tr=training_f1, f1_va='None' if validation_f1 is None else validation_f1,
            er_tr=training_er, er_va='None' if validation_er is None else validation_er,
            t=time_elapsed,
            l_f_spec=_loss_f_spec, acc_f_spec=_acc_f_spec, t_f_spec=_time_f_spec,
            e_spec=_epoch_f_spec)

    print_msg(the_msg, start='  -- ')


def print_evaluation_results(f1_score, er_score, time_elapsed):
    """Prints the output of the testing process.

    :param f1_score: The F1 score.
    :type f1_score: float
    :param er_score: The error rate.
    :type er_score: float
    :param time_elapsed: The elapsed time for the epoch.
    :type time_elapsed: float
    """
    the_msg = 'F1:{f1:{acc_f_spec}f} | ER:{er:{acc_f_spec}f} | Time:{t:{t_f_spec}f}'.format(
            f1=f1_score, er=er_score, t=time_elapsed,
            acc_f_spec=_acc_f_spec, t_f_spec=_time_f_spec)

    print_msg(the_msg, start='  -- ')


def print_modules_info(module, indent=2, start=''):
    """Prints modules names, weights, and biases.

    :param module: The module to use.
    :type module: torch.nn.Module
    :param indent: Amount of spaces to use as indentation.
    :type indent: int
    :param start: Starting decoration for the strings.
    :type start: str
    """
    weights_str = ': weights: {digits_w:{fmt_spec_w}d} | ' \
                  'biases: {digits_b:{fmt_spec_b}d}'

    w_and_bs = get_weights_and_biases_of_module(module)

    numel_weights, numel_biases = [], []

    for entry in w_and_bs:
        try:
            numel_weights.append(entry.weights_and_biases[0].numel())
            numel_biases.append(entry.weights_and_biases[1].numel())
        except TypeError:
            numel_weights.append(0)
            numel_biases.append(0)
        except AttributeError:
            numel_weights.append(sum([i.numel() for i in entry.weights_and_biases[0]]))
            numel_biases.append(sum([i.numel() for i in entry.weights_and_biases[1]]))

    max_len_name = max([len(entry.module_name) + entry.ind_level * indent for entry in w_and_bs])
    max_len_weights = max([len(str(w)) for w in numel_weights])
    max_len_biases = max([len(str(b)) for b in numel_biases])

    total_params = sum([sum([w, b]) for w, b in zip(numel_weights, numel_biases)])

    [print_msg(
        '{msg:{max_len}}{params}'.format(
            msg=entry.module_name,
            max_len=max_len_name - (entry.ind_level * indent),
            params=weights_str.format(
                fmt_spec_w=max_len_weights, fmt_spec_b=max_len_biases,
                digits_w=numel_weights[e], digits_b=numel_biases[e]
            ) if entry.weights_and_biases is not None else ''),
        start='{}{}'.format(start, ''.join([' ' * indent] * entry.ind_level))
    ) for e, entry in enumerate(w_and_bs)]
    print_msg('Total parameters: {}'.format(total_params), start='\n')


def get_weights_and_biases_of_module(module):
    """Gets the weights and biases of the module.

    :param module: The module.
    :type module: torch.nn.Module
    :return: A list of named tuples, each tuple having:
               - The name of the module (module_name).
               - Weights and biases or None `weights_biases`
               - Indentation level (ind_level)
    :rtype: list[collections.namedtuple]
    """
    Entry = namedtuple('Entry', ['module_name', 'weights_and_biases', 'ind_level'])

    entries = []
    modules_it = module.named_modules()

    i, n = next(modules_it)
    entries.append(Entry(
        module_name=n._get_name(),
        weights_and_biases=None,
        ind_level=0))

    for i, n in modules_it:
        if len(list(n.modules())) > 1:
            module_name = n._get_name()
            weights_and_biases = None
        else:
            module_name = str(n)
            try:
                weights_and_biases = [n.weight, n.bias]
            except AttributeError:
                try:
                    w = [getattr(n, w) for l in n._all_weights for w in l if w.startswith('weight')]
                    b = [getattr(n, w) for l in n._all_weights for w in l if w.startswith('bias')]
                    weights_and_biases = [w, b]
                except AttributeError:
                    weights_and_biases = None
        entries.append(Entry(
            module_name=module_name,
            weights_and_biases=weights_and_biases,
            ind_level=len(i.split('.'))))

    return entries


# EOF
