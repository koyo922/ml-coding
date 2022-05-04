#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
Factorization Machine at https://zhuanlan.zhihu.com/p/145436595

Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 10:17 AM
"""

from ml_coding.data import DatasetDiabetes

df_samples = DatasetDiabetes().fetch().train_test_split(0.2).df_samples
print(df_samples.shape)
