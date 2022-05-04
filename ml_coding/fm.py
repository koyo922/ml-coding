#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number
"""
Factorization Machine
Authors: qianweishuo<qianweishuo@bytedance.com>
Date:    2022/5/4 10:17 AM
"""
import numpy as np

from ml_coding.base_model import BaseModel
from ml_coding.util import init_log

logger = init_log(__name__)


# noinspection PyPep8Naming
class FM(BaseModel):
    def __init__(self):
        self.w_0, self.w_1, self.v = 0, None, None

    def fit(self, X, Y, k=4, num_epoch=500, eta=1e-3):
        m, n = X.shape
        self.w_0 = 0  # constant bias
        self.w_1 = np.random.normal(size=(n,))  # params for 1-order
        self.v = np.random.normal(0, 0.2, size=(n, k))  # params for 2-order

        for epoch in range(num_epoch):
            Y_hat = self.predict(X)  # 正则项比较麻烦，手写版本就省略了。在pytorch版本中比较好实现一些。
            d_L_yhat = (self.sigmoid(Y * Y_hat) - 1) * Y  # 每行不同方向和权重 (m,)
            self.w_0 -= eta * d_L_yhat.sum()
            self.w_1 -= (eta * d_L_yhat) @ X
            weight = (eta * d_L_yhat).reshape(-1, 1)
            self.v -= (X.T @ (X * weight) @ self.v  # 行看x, 列看inter
                       - np.diag(((X ** 2) * weight).sum(axis=0)) @ self.v)  # 二阶权重矩阵每行乘以 x^2

            if epoch % 10 == 0:
                loss = self.get_loss(self.predict(X), Y)
                logger.info("第 %d 次迭代后的损失为 %.2f", epoch, loss)
        return self

    def get_loss(self, Y_pred, Y_true):
        return -np.log(self.sigmoid(Y_pred * Y_true)).sum()

    def predict(self, X):
        inter_1 = X @ self.v  # (m,n)@(n,k) -> (m,k)
        inter_2 = (X ** 2) @ (self.v ** 2)  # (m,k)
        interaction = 1 / 2 * (inter_1 ** 2 - inter_2).sum(axis=1)  # (m,)
        y_hat = (self.w_0  # 常数bias
                 + X @ self.w_1  # (m,n)@(n,) -> (m,)
                 + interaction)  # (m,)
        return y_hat
