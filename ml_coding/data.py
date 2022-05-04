#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
get datasets

Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 10:28 AM
"""
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    uri: str

    def __init__(self):
        self.df_samples = pd.DataFrame()

    @property
    def pkl_path(self):
        return f'cache/{self.__class__.__name__}.pkl'

    def fetch(self, read_cache=True, write_cache=True):
        if read_cache and os.path.exists(self.pkl_path):
            self.df_samples = pd.read_pickle(self.pkl_path)
        elif self.uri.endswith('.csv'):
            self.df_samples = pd.read_csv(self.uri)
        else:
            raise NotImplementedError(f'---------- unsupported fetch() for object={repr(self)}')
        if write_cache and not os.path.exists(self.pkl_path):
            self.df_samples.to_pickle(self.pkl_path)
        return self

    def train_test_split(self, test_size=0.3):
        idx_trn, idx_tst = train_test_split(self.df_samples.index, test_size=test_size)
        self.df_samples.loc[idx_trn, 'is_train'] = 1
        self.df_samples.loc[idx_tst, 'is_train'] = 0
        return self


class DatasetDiabetes(Dataset):
    uri = 'https://github.com/susanli2016/Machine-Learning-with-Python/raw/master/diabetes.csv'
