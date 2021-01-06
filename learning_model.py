from model_selector.model_evaluator import ModelEvaluator


class LearningModel:

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
        """evaluate_model method to evalutate the model with many evaluation methods.

        model = LearningModel()
        adj_rs, mae = model.evaluate_model()

        :parameter
        none.

        :returns
        adj_r : double
            decimal value represent the R^2 error of the current model

        mae : double
            decimal value represent the mean absolute error of the current model
        """
        if self._x_validation is None:
            self._x_validation = self._x_train
        if self._y_validation is None:
            self._y_validation = self._y_train
        adj_rs = ModelEvaluator.adjust_r_squar_error(model=self._model, x_validation=self._x_validation,
                                                     y_validation=self._y_validation)
        mae = ModelEvaluator.get_mean_absolute_error(self._model, self._x_validation, self._y_validation)
        return adj_rs, mae

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    # get the name of the model as string
    def to_string(self):
        pass
