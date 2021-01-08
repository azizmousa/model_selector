import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score

class ModelEvaluator:

    @staticmethod
    def adjust_r_squar_error(model, x_validation, y_validation):
        y_hat = model.predict(x_validation)
        ss_resduals = sum((y_validation.flatten() - y_hat.flatten()) ** 2)
        ss_total = sum((y_validation.flatten() - np.mean(y_validation)) ** 2)
        r_squar = 1 - (float(ss_resduals) / ss_total)
        n = len(y_validation)
        p = len(x_validation[0])
        adjust_r_squar = 1 - (1 - r_squar) * ((n - 1) / np.abs(n - p - 1))
        return adjust_r_squar

    @staticmethod
    def get_mean_absolute_error(model, x_validation, y_validation):
        y_hat = model.predict(x_validation)
        return mean_absolute_error(y_validation, y_hat)

    @staticmethod
    def get_model_accuarcy(model, x_validation, y_validation):
        y_hat = model.predict(x_validation)
        return accuracy_score(y_validation, y_hat)
