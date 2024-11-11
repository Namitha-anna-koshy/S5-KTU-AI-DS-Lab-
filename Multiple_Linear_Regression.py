#Mutiple Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('multiple_linear_regression_data.csv')

x = df[['Feature1', 'Feature2']]
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

coefficients = model.coef_
intercept = model.intercept_
equation = f"y = {intercept:.2f} + {coefficients[0]:.2f}*Feature1 + {coefficients[1]:.2f}*Feature2"
print("Equation of the model:")
print(equation)
