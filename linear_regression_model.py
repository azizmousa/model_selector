from regression_model import RegressionModel
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(RegressionModel):

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None):
        if model is None:
            model = LinearRegression()
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        print("Training Linear Regression Model ....")
        self._model.fit(self._x_train, self._y_train)
        print("Linear Regression Model Training is Finished.>")

    # get the name of the model as string
    def to_string(self):
        return type(self._model)

