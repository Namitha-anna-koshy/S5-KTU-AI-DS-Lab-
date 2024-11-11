#Implementation of PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.DataFrame({'Age':[20,21,22,23,24,25],"Height":[170,172,169,166,168,171],'Weight':[65,60,53,55,59,50]})
scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)
pca=PCA(n_components=2)
pca_result=pca.fit_transform(scaled_data)

pca_df=pd.DataFrame(data=pca_result,columns=['P1',"P2"])
print(pca_df)

cov_mat=np.cov(pca_result)
print("Covariance Matrix of the pca result is : \n",cov_mat)

sns.heatmap(data)
plt.title("Original data")
plt.show()

sns.heatmap(pca_result)
plt.title('PCA Result')
plt.show()