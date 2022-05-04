#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 10:47 AM
"""
from sklearn import preprocessing

from ml_coding.base_model import timing
from ml_coding.data import DatasetDiabetes
from ml_coding.fm import FM
from ml_coding.util import init_log

logger = init_log(__name__)


def test_predict():
    x_trn, x_tst, y_trn, y_tst = (
        DatasetDiabetes().fetch()
            .train_test_split(0.2).split_x_y()
            # feature scaling after train/test split, to avoid future function
            .transform_x(preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform)
            .transform_y(lambda y: y * 2 - 1)  # 0/1 -> -1/+1
            .get_x_y()
    )

    fm = FM()

    with timing(label='training FM'):
        fm.fit(X=x_trn, Y=y_trn, k=4, num_epoch=100, eta=0.01)
        logger.info(f'params: w_0={fm.w_0}, w_1={fm.w_1}, v={fm.v}')

    y_pred = fm.predict(x_trn)
    logger.info(f"训练准确性为：{fm.get_accuracy(y_pred, y_trn)}")

    y_pred_tst = fm.predict(x_tst)  # 得到训练的准确性
    logger.info(f"测试准确性为：{fm.get_accuracy(y_pred_tst, y_tst)}")
