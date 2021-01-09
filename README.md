# >>>>>>>>>>> UNDER DEVELOPMENT <<<<<<<<<<<<

# >>>>>>> example for minimal regression selector

# from model_selector.regression.regression_selector import RegressionSelector
# form model_selector.model_evaluator import ModelEvaluator
# selector = RegressionSelector(x_train=x_train, y_train=y_train, x_validation=x_test, y_validation=y_test)
# selector.start_evaluation(ModelEvaluator.adjust_r_squar_error)
# print(selector.get_evaluation_array())
# best_model = selector.get_best_fit_model()
# print(f"Best Model Fit your Dataset is: {best_model[2]} with evaluation = {best_model[1]}")
#

# >>>>>>> example for minimal classification selector

# form model_selector.model_evaluator import ModelEvaluator
# from model_selector.classification.classification_selector import ClassificationSelector
# selector = ClassificationSelector(x_train, y_train.flatten(), x_test, y_test.flatten())
# selector.start_evaluation(ModelEvaluator.get_f1_score)
# print(selector.get_evaluation_array())
# best_model = selector.get_best_fit_model()
# print(f"Best Model Fit your Dataset is: {best_model[2]} with evaluation = {best_model[1]}")
#
