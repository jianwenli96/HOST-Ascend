# coding=utf-8
"""Get algorithm."""

from algos.alignment import Alignment

def get_algo(algo_type):
    if algo_type == 'alignment':
        return Alignment()
    else:
        raise ValueError('Algorithm %s not supported' % algo_type)
