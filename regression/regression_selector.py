import numpy as np
from model_selector.learning_model import LearningModel
from model_selector.regression.linear_regression_model import LinearRegressionModel
from model_selector.regression.polynomial_regression_model import PolynomialRegressionModel
from sklearn.preprocessing import StandardScaler
from model_selector.regression.support_vector_regression import SupportVectorRegression
from model_selector.regression.regression_tree_model import RegressionTreeModel
from model_selector.regression.random_forest_regression_model import RandomForestRegressionModel
from model_selector.model_evaluator import ModelEvaluator


class RegressionSelector:
    __models_evaluation = {}
    __x_train = None
    __y_train = None
    __x_validation = None
    __y_validation = None

    def __init__(self, x_train, y_train, x_validation=None, y_validation=None,
                 linear_model=None, polynomial_model=None, svr_model=None, random_forest_model=None,
                 regression_tree_model=None):
        self.__models_evaluation = {}
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
        return self.__models_evaluation.keys()

    def set_model(self, model):
        if not isinstance(model, LearningModel):
            raise TypeError("model should be type of LearningModel")
        if isinstance(model, PolynomialRegressionModel):
            self.__models_evaluation[model] = []
        else:
            self.__models_evaluation[model] = np.nan

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
        for model in self.__models_evaluation.keys():
            model.create_model()
            print(f"evaluating {type(model)} model .....")
            if isinstance(model, PolynomialRegressionModel):
                for eva in model.evaluate_model(ModelEvaluator.adjust_r_squar_error):
                    self.__models_evaluation[model].append(eva)
            else:
                self.__models_evaluation[model] = model.evaluate_model(ModelEvaluator.adjust_r_squar_error)

    def get_evaluation_array(self):
        return self.__models_evaluation.values()

    def get_best_fit_model(self):
        mx_model = None
        mx_model_type = ""
        i = 0
        mx = 0
        for model in self.__models_evaluation.keys():
            if isinstance(model, PolynomialRegressionModel):
                val = model.get_max_evaluation()
                if val[1] > mx:
                    mx = val[1]
                    mx_model = model.get_degreed_models()[val[0]]
                    mx_model_type = model.to_string(val[0])
            else:
                if self.__models_evaluation[model] > mx:
                    mx = self.__models_evaluation[model]
                    mx_model = model
                    mx_model_type = model.to_string()
            i += 1
        ret_val = (mx_model, mx, mx_model_type)
        return ret_val
