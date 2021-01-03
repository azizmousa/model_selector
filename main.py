import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from regression_selector import RegressionSelector


dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)

# clt = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
# X = clt.fit_transform(X)

x_tarin, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

selector = RegressionSelector(x_train=x_tarin, y_train=y_train, x_validation=x_test, y_validation=y_test)
selector.start_evaluation()
print(selector.get_evaluation_array())
best_model = selector.get_best_fit_model()
print(f"Best Model Fit your Dataset is: {type(best_model[0])} with adjust r^2 = {best_model[1]}")

