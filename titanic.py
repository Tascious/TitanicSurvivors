import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

data_train =pd.read_csv("python\\AICompetetion\\dataset\\train.csv")
data_test=pd.read_csv("python\\AICompetetion\\dataset\\test.csv")
df_train=pd.DataFrame(data_train)
df_test =pd.DataFrame(data_test)

#print(df_train.describe())
#print(df_train.isnull().sum())

median=df_train["Age"].median()
df_train["Age"]=df_train["Age"].fillna(median)
df_train["Cabin"]=df_train["Cabin"].fillna("C90")
df_train["Embarked"]=df_train["Embarked"].fillna("S")


#print(df_train.isnull().sum())
df_train.hist(figsize=(9,9),edgecolor="grey")
plt.show()

df_train.boxplot(figsize=(18,9))
plt.show()
df_train.groupby(["Survived"])[df_train.columns].mean().plot.bar()

xTrain=df_train.drop(columns=["Survived"])
yTrain=df_train["Survived"]
plt.figure(figsize=(8,6))
sns.heatmap(xTrain.corr(),annot=True,fmt='0.2f',cmap='YlGnBu')


med_test_age=df_test["Age"].median()
df_test["Age"]=df_test["Age"].fillna(med_test_age)
df_test["Cabin"]=df_test["Cabin"].fillna("C100")
med_test_fare=df_test["Fare"].median()
df_test["Fare"]=df_test["Fare"].fillna(med_test_fare)

#print(df_test.isnull().sum())

x_train,X_testn,y_train,y_testn=train_test_split(xTrain,yTrain,test_size=0,random_state=3 )#use ,stratify=yTrain in a case where i dont have another file for training
xTest=df_test
x_trainm,X_test,y_trainm,y_test=train_test_split(xTest,yTrain,test_size=1,random_state=2)

sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_train_scaled=pd.DataFrame(x_train_scaled,xTrain.columns)
X_test_scaled=sc.fit_transform(X_test)
X_test_scaled=pd.DataFrame(X_test_scaled,xTest.columns)

model_lgr=LogisticRegression()
model_lgr.fit(x_train_scaled,y_train)

y_pred_train=model_lgr.predict(x_train_scaled)
x_pred_test=model_lgr.predict(X_test_scaled)

print("Accuracy:",accuracy_score(y_train,y_pred_train))
print("Precision:",precision_score(y_train,y_pred_train))
print("Recall:",recall_score(y_train,y_pred_train))

cm =confusion_matrix(y_train,y_pred_train)
plt.figure(figsize=(6,4))
sns.heat(cm,annnot=True,fmt='.2f',xticklbels=['Not Survived','Survived'],yticklabels=['Not Survived','survived'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


