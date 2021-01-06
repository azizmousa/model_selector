from model_selector.regression.regression_evaluator import RegressionEvaluator


class RegressionModel:
    _x_train = None
    _y_train = None
    _x_validation = None
    _y_validation = None
    _model = None

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None):
        self._x_train = x_train
        self._y_train = y_train
        self._x_validation = x_validation
        self._y_validation = y_validation
        self._model = model

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        pass

    def evaluate_model(self):
        if self._x_validation is None:
            self._x_validation = self._x_train
        if self._y_validation is None:
            self._y_validation = self._y_train
        adj_rs = RegressionEvaluator.adjust_r_squar_error(model=self._model, x_validation=self._x_validation,
                                                          y_validation=self._y_validation)
        return adj_rs

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    # get the name of the model as string
    def to_string(self):
        pass
