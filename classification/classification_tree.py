from model_selector.learning_model import LearningModel
from sklearn.tree import DecisionTreeClassifier


class ClassificationTreeModel(LearningModel):

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None,
                 criterion='entropy'):
        self._criterion = criterion
        if model is None:
            model = DecisionTreeClassifier(criterion=criterion)
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        print("Training Classification Tree Model ....")
        self._model.fit(self._x_train, self._y_train)
        print("Classification Tree Model Training is Finished.>")

    # get the name of the model as string
    def to_string(self):
        return type(self._model)
