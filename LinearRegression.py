#Linear Regression

from sklearn.model_selection import train_test_split
from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('height_weight_data.csv')
x=df[' Height (Inches)']
y=df[' Weight (Pounds)']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

slope,intercept,r_value,p_value,std_err=linregress(x_test,y_test)
y_pred=slope*x_test+intercept

print(f'Covariance :{np.cov(x,y)[0,1]:.3f}')

plt.scatter(x_test,y_test,color='red',label='Actual data')
plt.plot(x_test,y_pred,color='black',label='Linear Regression')
plt.xlabel("Height")
plt.ylabel('Weight'
plt.title("Simple Linear Regression"))
plt.legend()#to increase the clarity and readability
plt.show()


