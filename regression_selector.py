import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression_model import LinearRegressionModel
from polynomial_regression_model import PolynomialRegressionModel
from sklearn.preprocessing import StandardScaler
from support_vector_regression import SupportVectorRegression
from regression_tree_model import RegressionTreeModel
from random_forest_regression_model import RandomForestRegressionModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class RegressionSelector:
    __models = []

    def get_all_models(self):
        return self.__models

    def set_linear_model(self, model):
        if not isinstance(model, LinearRegressionModel):
            raise TypeError("model should be type of LinearRegression")
        self.__models.append(model)

    def set_polynomial_model(self, model):
        if not isinstance(model, PolynomialRegressionModel):
            raise TypeError("model should be type of LinearRegression")
        self.__models.append(model)

    def set_svr_model(self, model):
        if not isinstance(model, SupportVectorRegression):
            raise TypeError("model should be type of SVR")
        self.__models.append(model)

    def set_random_forest_model(self, model):
        if not isinstance(model, RandomForestRegressionModel):
            raise TypeError("model should be type of RandomForestRegressor")
        self.__models.append(model)

    def set_regression_tree_model(self, model):
        if not isinstance(model, RegressionTreeModel):
            raise TypeError("model should be type of DecisionTreeRegressor")
        self.__models.append(model)
