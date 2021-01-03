import numpy as np
from regression_model import RegressionModel
from linear_regression_model import LinearRegressionModel
from polynomial_regression_model import PolynomialRegressionModel
from sklearn.preprocessing import StandardScaler
from support_vector_regression import SupportVectorRegression
from regression_tree_model import RegressionTreeModel
from random_forest_regression_model import RandomForestRegressionModel


class RegressionSelector:
    __models = []
    __adjust_r_square_arr = []
    __x_train = None
    __y_train = None
    __x_validation = None
    __y_validation = None

    def __init__(self, x_train, y_train, x_validation=None, y_validation=None,
                 linear_model=None, polynomial_model=None, svr_model=None, random_forest_model=None,
                 regression_tree_model=None):
        self.__models = []
        self.__adjust_r_square_arr = []
        self.__x_train = x_train
        self.__x_validation = x_validation
        self.__y_train = y_train
        self.__y_validation = y_validation

        if linear_model is None:
            linear_model = LinearRegressionModel(x_train=x_train, y_train=y_train,
                                                 x_validation=x_validation, y_validation=y_validation)

        if polynomial_model is None:
            polynomial_model = PolynomialRegressionModel(x_train=x_train, y_train=y_train, x_validation=x_validation,
                                                         y_validation=y_validation, degree_range=np.arange(1, 5))

        if svr_model is None:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            svr_model = SupportVectorRegression(x_train=x_train, y_train=y_train, x_validation=x_validation,
                                                y_validation=y_validation, x_scaler=x_scaler, y_scaler=y_scaler)

        if random_forest_model is None:
            random_forest_model = RandomForestRegressionModel(x_train=x_train, y_train=y_train,
                                                              x_validation=x_validation, y_validation=y_validation,
                                                              n_estimators=10)

        if regression_tree_model is None:
            regression_tree_model = RegressionTreeModel(x_train=x_train, y_train=y_train, x_validation=x_validation,
                                                        y_validation=y_validation)
        self.set_model(linear_model)
        self.set_model(polynomial_model)
        self.set_model(svr_model)
        self.set_model(random_forest_model)
        self.set_model(regression_tree_model)

    def get_all_models(self):
        return self.__models

    def set_model(self, model):
        if not isinstance(model, RegressionModel):
            raise TypeError("model should be type of RegressionModel")
        self.__models.append(model)

    def set_x_trainset(self, x_train):
        self.__x_train = x_train

    def set_y_trainset(self, y_train):
        self.__y_train = y_train

    def set_x_validationset(self, x_validation):
        self.__x_validation = x_validation

    def set_y_validationset(self, y_validation):
        self.__y_validation = y_validation

    def get_x_trainset(self):
        return self.__x_train

    def get_y_trainset(self):
        return self.__y_train

    def get_x_validationset(self):
        return self.__x_validation

    def get_y_validationset(self):
        return self.__y_validation
