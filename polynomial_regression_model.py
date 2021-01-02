from regression_model import RegressionModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PolynomialRegressionModel(RegressionModel):

    _degree_range = []
    _degreed_model = {}

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, degree_range=None):
        if degree_range is None:
            degree_range = []
        self._degree_range = degree_range
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

    # get the name of the model as string
    def to_string(self):
        return type(self._model)
