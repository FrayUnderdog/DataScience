import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Input Dateset
org_df = pd.read_csv("house_ds.csv")

#Define features and outcome for Regression
outcome_df =  org_df.loc[:,org_df.columns == 'price']
feat_df =  org_df.loc[:,org_df.columns == 'sqft_living']

#Log Transform
outcome_df['price'] = np.log(outcome_df['price'])
feat_df['sqft_living'] = np.log(feat_df['sqft_living'])

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,outcome_df,test_size=0.25)

#Create a Reg model
model = LinearRegression()
model.fit(train_x,  train_y)
print('slope=',model.coef_)
print('intercept=',model.intercept_)

#test_pred_y = model.coef_ * test_x + model.intercept_
test_pred_y = model.predict(test_x)

#Visualize the model
plt.scatter(test_x, test_y)
plt.plot(test_x, test_pred_y)
plt.show()

#test_accuracy=accuracy_score(test_pred_y,test_y)
r_sq = model.score(test_x, test_y)
print ('R2 =',r_sq )

##############################################Multiple Reg

#Input Dateset
org_df = pd.read_csv("house_ds.csv")

#Define features and outcome for Regression
outcome_df =  org_df.loc[:,org_df.columns == 'price']
feat_df =  org_df.loc[:,org_df.columns != 'price']

#Log Transform
outcome_df['price'] = np.log(outcome_df['price'])
feat_df['sqft_living'] = np.log(feat_df['sqft_living'])

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,outcome_df,test_size=0.25)

#Create a multiple Reg model
model = LinearRegression()
model.fit(train_x,  train_y)

test_pred_y = model.predict(test_x)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(train_x['sqft_living'], train_x['yr_built'], train_y['price'], marker="*")
ax.set_xlabel('sqft_living')
ax.set_ylabel('yr_built')
ax.set_zlabel('price')
plt.show()

r_sq = model.score(test_x, test_y)
print ('R2 =',r_sq )

########################################################

#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Define features and label for KNN
label_df =  org_df.loc[:,org_df.columns == 'Outcome']
feat_df =  org_df.loc[:,org_df.columns != 'Outcome']

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,label_df,test_size=0.25)

#KNN Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x, train_y)

test_pred_y = knn.predict(test_x)

# accuracy
accuracy = accuracy_score(test_y, test_pred_y)
print( "Accuracy=", accuracy)
