#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
general base class for models, including common components e.g. softmax()

Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 11:22 AM
"""
import contextlib

import arrow
import numpy as np


class BaseModel:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


class TimingResult:
    def __init__(self, label, last=None):
        self.label = label  # type: str
        self.last = last  # type timedelta


@contextlib.contextmanager
def timing(label='unnamed-task', log_fn=print, silent_start=True):
    if log_fn is not None and not silent_start:
        log_fn('[STRT] {}'.format(label))
    start = arrow.now()
    t = TimingResult(label=label)
    yield t
    t.last = arrow.now() - start
    if log_fn is not None:
        log_fn('[DONE] {} 耗时: {}'.format(label, t.last))
