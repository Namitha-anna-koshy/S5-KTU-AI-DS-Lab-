#KNN CLASSIFIER
#import necessary libraries
from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#Load data set
iris=load_iris()
X=iris.data
y=iris.target
#Split the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=45)
#Train the model using knn classifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
#Make predictions
pred=knn.predict(X_test)
#Evaluate the model
print("Áccuracy Score : ",accuracy_score(y_test,pred))
print("CLassification Report : ",classification_report(y_test,pred))
#Correct and wrong predictions
print('Correct predictions :')
for i in range (len(y_test)):
    if y_test[i]==pred[i]:
        print(f'Index : {i},Actual : {y_test[i]} Predicted : {pred[i]}')
print('Wrong predictions :')
for i in range (len(y_test)):
    if y_test[i]!=pred[i]:
        print(f'Index : {i},Actual : {y_test[i]} Predicted : {pred[i]}')