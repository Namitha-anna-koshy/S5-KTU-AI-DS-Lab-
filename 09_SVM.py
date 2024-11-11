#Implementation of SVM

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("HeartDisease.csv")
x=df[['cp','trestbps','chol','fbs']]
y=df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Accuracy : ",accuracy_score(y_test,y_pred))
print('Classification Report : ',classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Disease','Disease'])
display.plot(cmap=plt.cm.Blues)
plt.show()
