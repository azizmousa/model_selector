from regression_model import RegressionModel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from regression_evaluator import RegressionEvaluator


class PolynomialRegressionModel(RegressionModel):

    _degree_range = []
    _degreed_model = {}

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, degree_range=None):

        self.set_degree_range(degree_range)
        self._degreed_model = {}
        if model is None:
            model = LinearRegression()
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        if self._degree_range is []:
            print("Training Polynomial Regression Model of Degree [1] ....")
            self._model.fit(self._x_train, self._y_train)
        else:
            for degree in self._degree_range:
                print(f"Training Polynomial Regression Model of Degree [{degree}] ....")
                poly = PolynomialFeatures(degree=degree)
                x_poly_train = poly.fit_transform(self._x_train)
                model = LinearRegression()
                model.fit(x_poly_train, self._y_train)
                self._degreed_model[degree] = model
                print(f"Polynomial Regression Model of Degree [{degree}] Training is Finished >>")

    def evaluate_model(self):
        if self._x_validation is None:
            self._x_validation = self._x_train
        if self._y_validation is None:
            self._y_validation = self._y_train
        for degree in self._degreed_model:
            poly = PolynomialFeatures(degree=degree)
            adj_rs = RegressionEvaluator.adjust_r_squar_error(model=self._degreed_model[degree],
                                                              x_validation=poly.fit_transform(self._x_validation),
                                                              y_validation=self._y_validation)
            yield adj_rs

    # get the name of the model as string
    def to_string(self):
        return type(self._model)

    def get_degree_range(self):
        return self._degree_range

    def get_degreed_models(self):
        return self._degreed_model

    def set_degree_range(self, degree_range):
        if degree_range is None:
            degree_range = []
        elif not isinstance(degree_range, np.ndarray):
            raise TypeError("degree_range should be 1d array")
        self._degree_range = degree_range.flatten()
