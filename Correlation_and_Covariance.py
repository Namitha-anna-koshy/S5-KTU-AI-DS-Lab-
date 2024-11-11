#Finding Correlation and Covariance

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

data=load_iris()
df=pd.DataFrame(data=data,columns=data.feature_names)

correlation_matrix=df.corr()
covariance_matrix=df.cov()

correlation_matrix.to_csv('correlation_matrix.csv', index=True)
covariance_matrix.to_csv('covariance_matrix.csv', index=True)
print("CSV files created successfully.")


print("Correlation matrix : \n",correlation_matrix)
print('Covariance matrix : \n',covariance_matrix)