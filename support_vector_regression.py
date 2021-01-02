from regression_model import RegressionModel
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SupportVectorRegression(RegressionModel):

    _x_scaler = None
    _y_scaler = None

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, x_scaler=None,
                 y_scaler=None):

        self._x_scaler = x_scaler
        self._y_scaler = y_scaler

        if model is None:
            model = SVR(kernel="rbf")
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        print("Training Support Vector Regression Model ....")
        x_train_tmp = self._x_train
        y_train_tmp = self._y_train

        if self._x_scaler is not None:
            x_train_tmp = self._x_scaler.fit_transform(x_train_tmp)
        if self._y_scaler is not None:
            y_train_tmp = self._y_scaler.fit_transform(y_train_tmp)

        self._model.fit(x_train_tmp, y_train_tmp.flatten())
        print("Support Vector Regression Model Training is Finished.>")

    # get the name of the model as string
    def to_string(self):
        return type(self._model)
