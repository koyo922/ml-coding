#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
get datasets

Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 10:28 AM
"""
import os.path

# noinspection PyUnresolvedReferences
import janitor
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    uri: str
    label_col: str

    def __init__(self):
        self.df_samples = pd.DataFrame()
        self.has_split_x_y = False
        self.has_split_trn_tst = False
        self.X, self.y = None, None
        self.df_trn, self.df_tst = None, None
        self.X_trn, self.y_trn, self.X_tst, self.y_tst = None, None, None, None

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
        assert not self.has_split_trn_tst
        idx_trn, idx_tst = train_test_split(self.df_samples.index, test_size=test_size)
        if self.has_split_x_y:
            self.X_trn, self.X_tst = self.X.loc[idx_trn], self.X.loc[idx_tst]
            self.y_trn, self.y_tst = self.y.loc[idx_trn], self.y.loc[idx_tst]
        else:
            self.df_trn, self.df_tst = self.df_samples.loc[idx_trn], self.df_samples.loc[idx_tst]
        self.has_split_trn_tst = True
        return self

    def split_x_y(self):
        assert not self.has_split_x_y
        if self.has_split_trn_tst:
            self.X_trn, self.y_trn = self.df_trn.get_features_targets(target_column_names=self.label_col)
            self.X_tst, self.y_tst = self.df_tst.get_features_targets(target_column_names=self.label_col)
        else:
            self.X, self.y = self.df_samples.get_features_targets(target_column_names=self.label_col)
        self.has_split_x_y = True
        return self

    def transform_x(self, fn):
        assert self.has_split_x_y
        if self.has_split_trn_tst:
            self.X_trn, self.X_tst = fn(self.X_trn), fn(self.X_tst)
        else:
            self.X = fn(self.X)
        return self

    def transform_y(self, fn):
        assert self.has_split_x_y
        if self.has_split_trn_tst:
            self.y_trn, self.y_tst = fn(self.y_trn), fn(self.y_tst)
        else:
            self.y = fn(self.y)
        return self

    def get_x_y(self):
        assert self.has_split_x_y
        if self.has_split_trn_tst:
            return self.X_trn, self.X_tst, self.y_trn, self.y_tst
        else:
            return self.X, self.y


class DatasetDiabetes(Dataset):
    uri = 'https://github.com/susanli2016/Machine-Learning-with-Python/raw/master/diabetes.csv'
    label_col = 'Outcome'
