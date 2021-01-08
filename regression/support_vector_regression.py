from model_selector.learning_model import LearningModel
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


class SupportVectorRegression(LearningModel):
    _x_scaler = None
    _y_scaler = None

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, x_scaler=None,
                 y_scaler=None):

        self.set_x_scaler(x_scaler)
        self.set_y_scaler(y_scaler)

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

    def evaluate_model(self, eval_func):
        if self._x_validation is None:
            self._x_validation = self._x_train
        if self._y_validation is None:
            self._y_validation = self._y_train
        x_scaled = self._x_validation
        y_scaled = self._y_validation
        if self._x_scaler is not None:
            self._x_validation = self._x_scaler.fit_transform(x_scaled)

        if self._y_scaler is not None:
            self._y_validation = self._y_scaler.fit_transform(y_scaled)

        return super().evaluate_model(eval_func)

    # get the name of the model as string
    def to_string(self):
        return type(self._model)

    def get_x_scaler(self):
        return self._x_scaler

    def get_y_scaler(self):
        return self._y_scaler

    def set_x_scaler(self, scaler):
        if scaler is not None and not (isinstance(scaler, StandardScaler) or isinstance(scaler, MaxAbsScaler)
                                       or isinstance(scaler, MinMaxScaler) or isinstance(scaler, RobustScaler)):
            raise TypeError("type of scaler should be one of the scaler classes only")

        self._x_scaler = scaler

    def set_y_scaler(self, scaler):
        if scaler is not None and not (isinstance(scaler, StandardScaler) or isinstance(scaler, MaxAbsScaler)
                                       or isinstance(scaler, MinMaxScaler) or isinstance(scaler, RobustScaler)):
            raise TypeError("type of scaler should be one of the scaler classes only")

        self._y_scaler = scaler
