import numpy as np


class RegressionEvaluator:

    @staticmethod
    def adjust_r_squar_error(model, x_validation, y_validation):
        y_hat = model.predict(x_validation)
        ss_resduals = sum((y_validation.flatten() - y_hat.flatten()) ** 2)
        ss_total = sum((y_validation.flatten() - np.mean(y_validation)) ** 2)
        r_squar = 1 - (float(ss_resduals)/ss_total)
        n = len(y_validation)
        p = len(x_validation[0])
        adjust_r_squar = 1 - (1 - r_squar)*((n-1) / (n-p-1))
        return adjust_r_squar
