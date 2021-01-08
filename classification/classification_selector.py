import numpy as np
from model_selector.learning_model import LearningModel
from model_selector.classification.logistic_regression import LogisticRegressionModel
from model_selector.classification.k_nearset_neighbors import KNearestNeighborsModel
from model_selector.classification.support_vector_machine import SupportVectorMachineModel
from model_selector.classification.naive_bayes import NaiveBayesModel
from model_selector.classification.classification_tree import ClassificationTreeModel
from model_selector.classification.random_forest_classification_model import RandomForestClassificationModel
from model_selector.regression.polynomial_regression_model import PolynomialRegressionModel
from model_selector.model_evaluator import ModelEvaluator


class ClassificationSelector:
    __models_evaluation = {}
    __x_train = None
    __y_train = None
    __x_validation = None
    __y_validation = None

    def __init__(self, x_train, y_train, x_validation=None, y_validation=None,
                 logistic_model=None, knn_model=None, naive_model=None, random_forest_model=None,
                 classification_tree_model=None, svm_model=None):
        self.__models_evaluation = {}
        self.__x_train = x_train
        self.__x_validation = x_validation
        self.__y_train = y_train
        self.__y_validation = y_validation

        if logistic_model is None:
            logistic_model = LogisticRegressionModel(x_train, y_train, x_validation, y_validation)
        if knn_model is None:
            knn_model = KNearestNeighborsModel(x_train, y_train, x_validation, y_validation)
        if naive_model is None:
            naive_model = NaiveBayesModel(x_train, y_train, x_validation, y_validation)
        if random_forest_model is None:
            random_forest_model = RandomForestClassificationModel(x_train, y_train, x_validation, y_validation)
        if classification_tree_model is None:
            classification_tree_model = ClassificationTreeModel(x_train, y_train, x_validation, y_validation)
        if svm_model is None:
            svm_model = SupportVectorMachineModel(x_train, y_train, x_validation, y_validation)

        self.set_model(logistic_model)
        self.set_model(knn_model)
        self.set_model(naive_model)
        self.set_model(random_forest_model)
        self.set_model(classification_tree_model)
        self.set_model(svm_model)

    def get_all_models(self):
        return self.__models_evaluation.keys()

    def set_model(self, model):
        if not isinstance(model, LearningModel):
            raise TypeError("model should be type of LearningModel")
        self.__models_evaluation[model] = -1


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
            self.__models_evaluation[model] = model.evaluate_model(ModelEvaluator.get_model_accuarcy)

    def get_evaluation_array(self):
        return self.__models_evaluation.values()

    def get_best_fit_model(self):
        mx_model = None
        mx_model_type = ""
        mx = 0
        for model in self.__models_evaluation.keys():
            evl = self.__models_evaluation[model]
            if evl > mx:
                mx = self.__models_evaluation[model]
                mx_model = model
                mx_model_type = model.to_string()
        ret_val = (mx_model, mx, mx_model_type)
        return ret_val
