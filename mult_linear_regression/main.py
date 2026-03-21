import pandas
from pandas.plotting import scatter_matrix
import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

housing_dataset = pandas.read_csv("./mult_linear_regression/housing_dataset.csv")

print(housing_dataset.head())

matplotlib.rc("axes", labelsize=14)
matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)

print(housing_dataset.shape)

print(housing_dataset.info())

print(set(housing_dataset["ocean_proximity"]))

print(housing_dataset["ocean_proximity"].value_counts())

print(housing_dataset.describe())

# dataset.hist(bins=50, figsize=(15, 10))
# plt.show()

dataset_train, dataset_test = train_test_split(housing_dataset, test_size=0.2, random_state=47)

print(len(dataset_train), "train +", len(dataset_test), "test")

# dataset["median_income"].hist()
# plt.show()

housing_dataset["income_category"] = numpy.ceil(housing_dataset["median_income"] / 1.5)

print(housing_dataset["income_category"].value_counts())

housing_dataset["income_category"] = housing_dataset["income_category"].where(housing_dataset["income_category"] < 5, 5.0)

print(housing_dataset["income_category"].value_counts())

housing_dataset["income_category"] = pandas.cut(
  housing_dataset["median_income"], 
  bins=[0., 1.5, 3.0, 4.5, 6., numpy.inf], 
  labels=[1, 2, 3, 4, 5]
)

print(housing_dataset["income_category"].value_counts())

# dataset["income_category"].hist()
# plt.show()

stratified_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=47)

for train_index, test_index in stratified_shuffle_split.split(housing_dataset, housing_dataset["income_category"]):
  sss_dataset_train = housing_dataset.loc[train_index]
  sss_dataset_test = housing_dataset.loc[test_index]


print("Test proportions:\n", sss_dataset_test["income_category"].value_counts() / len(sss_dataset_test))
print("Train proportions:\n", sss_dataset_train["income_category"].value_counts() / len(sss_dataset_train))
print("Overall proportions:\n", housing_dataset["income_category"].value_counts() / len(housing_dataset))

for sss_dataset in (sss_dataset_train, sss_dataset_test):
  sss_dataset.drop("income_category", axis=1, inplace=True)

housing_train_dataset = sss_dataset_train.copy()

# housing_train_dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# housing_train_dataset.plot(
#   kind="scatter", 
#   x="longitude", 
#   y="latitude", 
#   alpha=0.4, 
#   s=housing_train_dataset["population"] / 100, 
#   label="population", 
#   figsize=(10, 7), 
#   c="median_house_value", 
#   cmap=plt.get_cmap("jet"), 
#   colorbar=True,
# )

# plt.show()

# corr_matrix = housing_train_dataset.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

columns = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# scatter_matrix(housing_train_dataset[columns], figsize=(12, 8))

# housing_train_dataset.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.axis([0, 16, 0, 550000])
# plt.show()

housing_train_dataset = sss_dataset_train.drop("median_house_value", axis=1)
housing_train_labels = sss_dataset_train["median_house_value"].copy()

# print(housing_train_dataset)
# print(housing_train_labels)

sample_incomplete_rows = housing_train_dataset[housing_train_dataset.isnull().any(axis=1)].head()
print(sample_incomplete_rows)

print("Sum of null values in each column:\n", housing_train_dataset.isnull().sum())

total_bedrooms_median = housing_train_dataset["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(total_bedrooms_median, inplace=True)
print(sample_incomplete_rows)

simple_imputer = SimpleImputer(strategy="median")

housing_train_dataset_numbers = housing_train_dataset.drop("ocean_proximity", axis=1)

simple_imputer.fit(housing_train_dataset_numbers)

print(simple_imputer.statistics_)

transformed_dataset = simple_imputer.transform(housing_train_dataset_numbers)

housing_train_dataset_treated = pandas.DataFrame(
  transformed_dataset, 
  columns=housing_train_dataset_numbers.columns,
  index= housing_train_dataset_numbers.index
)

print(housing_train_dataset_treated.head())

print(simple_imputer.strategy)

housing_train_dataset_categorical = housing_train_dataset[["ocean_proximity"]]

# print(housing_train_dataset_categorical.head(10))

ordinal_encoder = OrdinalEncoder()
housing_train_dataset_categorical_encoded = ordinal_encoder.fit_transform(housing_train_dataset_categorical)
print(housing_train_dataset_categorical_encoded[:10])
print(ordinal_encoder.categories_)

one_hot_encoder = OneHotEncoder(sparse_output=False)
housing_train_dataset_categorical_one_hot = one_hot_encoder.fit_transform(housing_train_dataset_categorical)
print(housing_train_dataset_categorical_one_hot[:10])
print(one_hot_encoder.categories_)


number_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy="median")),
  ('scaler', StandardScaler())
])

housing_train_dataset_numbers_treated = number_pipeline.fit_transform(housing_train_dataset_numbers)

attributes_numbers = housing_train_dataset_numbers.columns
attributes_categorical = ["ocean_proximity"]


full_pipeline = ColumnTransformer([
  ("numbers", number_pipeline, attributes_numbers),
  ("categorical", OneHotEncoder(), attributes_categorical)
])

housing_train_dataset_treated = full_pipeline.fit_transform(housing_train_dataset)

print(housing_train_dataset_treated.shape)

column_names = list(attributes_numbers) + list(full_pipeline.named_transformers_["categorical"].get_feature_names_out(attributes_categorical))

print(column_names)

housing_train_dataset_treated_df = pandas.DataFrame(housing_train_dataset_treated, columns=column_names)

print(housing_train_dataset_treated_df.shape)
print(housing_train_dataset_treated_df.head())
print("Sum of null values in each column:\n", housing_train_dataset_treated_df.isnull().sum())


linear_regression_model = LinearRegression()
linear_regression_model.fit(housing_train_dataset_treated, housing_train_labels)

housing_test_dataset = sss_dataset_test.drop("median_house_value", axis=1)
housing_test_labels = sss_dataset_test["median_house_value"].copy()

housing_test_dataset_treated = full_pipeline.transform(housing_test_dataset)

predictions = linear_regression_model.predict(housing_test_dataset_treated)

print(predictions[:10])
print(housing_test_labels[:10].values)


mse = mean_squared_error(housing_test_labels, predictions)
rmse = numpy.sqrt(mse)
mae = mean_absolute_error(housing_test_labels, predictions)
r2 = r2_score(housing_test_labels, predictions)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

def mean_absolute_percentage_error(labels, predictions):
  error = numpy.abs((labels - predictions) / labels)
  return numpy.mean(error) * 100

mape = mean_absolute_percentage_error(housing_test_labels, predictions)
print(f"Mean Absolute Percentage Error: {mape:.2f}%")