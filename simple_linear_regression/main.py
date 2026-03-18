import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

data = pd.read_excel('simple_linear_regression/ice_cream_dataset.xlsx')

pyplot.scatter(data['temperature'], data['sales']);
pyplot.xlabel('Temperature (°C)')
pyplot.ylabel('Ice Cream Sales (thousands)')
pyplot.title('Ice Cream Sales vs Temperature')
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(data[['temperature']], data['sales'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

pyplot.scatter(X_test, y_test, color='blue', label='Actual Sales')
pyplot.plot(X_test, y_prediction, color='red', label='Predicted Sales')
pyplot.xlabel('Temperature (°C)')
pyplot.ylabel('Ice Cream Sales (thousands)')
pyplot.title('Actual vs Predicted Ice Cream Sales')
pyplot.legend()
pyplot.show()


MSE = mean_squared_error(y_test, y_prediction)
MAE = mean_absolute_error(y_test, y_prediction)
R2 = r2_score(y_test, y_prediction)

print(f'Mean Squared Error: {MSE}')
print(f'Mean Absolute Error: {MAE}')
print(f'R-squared: {R2}')