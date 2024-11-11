#Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

importances = np.mean(np.abs(model.coef_),axis=0)

features = data.feature_names
importance_df = pd.DataFrame(importances, index=features, columns=['Importance'])
importance_df = importance_df.sort_values(by='Importance')
print('Feature Importance:')
print(importance_df)
print('Model Accuracy:', model.score(x_test_scaled, y_test))
