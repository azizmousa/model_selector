import numpy as np


class RegressionEvaluator:

    @staticmethod
    def adjust_r_squar_error(self, model, x_train, y_train):
        y_hat = model.predicit(x_train)
        ss_resduals = sum((y_train - y_hat)**2)
        ss_total = sum((y_train - np.mean(y_train))**2)
        r_squar = 1 - (float(ss_resduals)/ss_total)
        n = len(y_train)
        p = len(x_train[0])
        adjust_r_squar = 1 - (1 - r_squar)*((n-1) / (n-p-1))
        return adjust_r_squar
