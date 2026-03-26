import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mape import mean_absolute_percentage_error

insurance_dataset = pd.read_csv("./challenger_01/insurance.csv")

# print("Null columns:\n", insurance_dataset.isnull().sum())

# print(insurance_dataset.info())

# print(insurance_dataset.describe())

# print(insurance_dataset["sex"].value_counts())
# print(insurance_dataset["smoker"].value_counts())
# print(insurance_dataset["region"].value_counts())

insurance_train_dataset, insurance_test_dataset = train_test_split(
  insurance_dataset, test_size=0.2, random_state=47
)

insurance_train_dataset_labels = insurance_train_dataset["charges"].copy()
insurance_train_dataset.drop("charges", axis=1, inplace=True)

insurance_test_dataset_labels = insurance_test_dataset["charges"].copy()
insurance_test_dataset.drop("charges", axis=1, inplace=True)

numeric_features = ["age", "bmi", "children"]
text_features = ["sex", "smoker", "region"]

pipeline = ColumnTransformer([
  ('scaler', StandardScaler(), numeric_features),
  ('encoder', OrdinalEncoder(), text_features)
])

insurance_train_dataset_treated = pipeline.fit_transform(insurance_train_dataset)
insurance_test_dataset_treated = pipeline.transform(insurance_test_dataset)

columns = numeric_features + text_features

insurance_train_dataset_df = pd.DataFrame(insurance_train_dataset_treated, columns=columns)
insurance_test_dataset_df = pd.DataFrame(insurance_test_dataset_treated, columns=columns)

linear_regression = LinearRegression()
linear_regression.fit(insurance_train_dataset_df, insurance_train_dataset_labels)
linear_regression_predictions = linear_regression.predict(insurance_test_dataset_df)

MSE = mean_squared_error(insurance_test_dataset_labels, linear_regression_predictions)
RMSE = numpy.sqrt(MSE)
MAE = mean_absolute_error(insurance_test_dataset_labels, linear_regression_predictions)
R2 = r2_score(insurance_test_dataset_labels, linear_regression_predictions)
MAPE = mean_absolute_percentage_error(insurance_test_dataset_labels, linear_regression_predictions)

print("Mean Squared Error:", MSE)
print("Root Mean Squared Error:", RMSE)
print("Mean Absolute Error:", MAE)
print(f"Mean Absolute Percentage Error: {MAPE:.2f}%")
print(f"R2 Score: {R2:.2f} %")