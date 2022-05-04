#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 10:47 AM
"""
from ml_coding.data import DatasetDiabetes


def test_predict():
    df_samples = DatasetDiabetes().fetch().train_test_split(0.2).df_samples
    print(df_samples.shape)
    assert False
