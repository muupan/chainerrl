import collections

import chainer
import numpy as np


def max_or_none(xs):
    if xs:
        return np.max(xs)
    else:
        return np.nan


def min_or_none(xs):
    if xs:
        return np.min(xs)
    else:
        return np.nan


class StatisticsRecorder(object):

    def __init__(self):
        self.records = {}
        self.statistic_types = {}
        self.counts = {}

    def register(self, key, maxlen=100,
                 statistic_types=[
                     'population',
                     'mean',
                     'std',
                     'max',
                     'min',
                     'count']):
        assert key not in self.records
        self.records[key] = collections.deque(maxlen=maxlen)
        self.statistic_types[key] = statistic_types
        self.counts[key] = 0

    def record(self, key, value):
        if np.isscalar(value):
            self.records[key].append(value)
            self.counts[key] += 1
        else:
            self.records[key].extend(chainer.cuda.to_cpu(value.ravel()))
            self.counts[key] += value.size

    def get_statistics(self):
        keys = sorted(self.records)
        statistics = []
        for key in keys:
            for stat_type in self.statistic_types[key]:
                stat_key = key + '/' + stat_type
                stat_val = {
                    'population': lambda: len(self.records[key]),
                    'mean': lambda: np.mean(self.records[key]),
                    'std': lambda: np.std(self.records[key]),
                    'max': lambda: max_or_none(self.records[key]),
                    'min': lambda: min_or_none(self.records[key]),
                    'count': lambda: self.counts[key],
                }[stat_type]()
                statistics.append((stat_key, stat_val))
        return statistics
