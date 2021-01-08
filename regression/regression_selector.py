import numpy as np
from model_selector.learning_model import LearningModel
from model_selector.regression.linear_regression_model import LinearRegressionModel
from model_selector.regression.polynomial_regression_model import PolynomialRegressionModel
from sklearn.preprocessing import StandardScaler
from model_selector.regression.support_vector_regression import SupportVectorRegression
from model_selector.regression.regression_tree_model import RegressionTreeModel
from model_selector.regression.random_forest_regression_model import RandomForestRegressionModel


class RegressionSelector:
    __models = []
    __evaluation_arr = []
    __x_train = None
    __y_train = None
    __x_validation = None
    __y_validation = None

    def __init__(self, x_train, y_train, x_validation=None, y_validation=None,
                 linear_model=None, polynomial_model=None, svr_model=None, random_forest_model=None,
                 regression_tree_model=None):
        self.__models = []
        self.__evaluation_arr = []
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
        if not isinstance(model, LearningModel):
            raise TypeError("model should be type of LearningModel")
        if isinstance(model, PolynomialRegressionModel):
            for mdl in model.get_degreed_models():
                self.__models.append(mdl)
        else:
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

    def start_evaluation(self):
        for model in self.__models:
            model.create_model()
            print(f"evaluating {type(model)} model .....")
            if isinstance(model, PolynomialRegressionModel):
                for eva in model.evaluate_model():
                    self.__evaluation_arr.append(eva)
            else:
                self.__evaluation_arr.append(model.evaluate_model())

    def get_evaluation_array(self):
        return self.__evaluation_arr

    def get_best_fit_model(self):
        mx_model = None
        i = 0
        mx = (0, 0)
        for ev in self.__evaluation_arr:
            if ev[0] > mx[0]:
                mx = ev
                mx_model = self.__models[i]
            i += 1
        ret_val = (mx_model, mx)
        return ret_val
