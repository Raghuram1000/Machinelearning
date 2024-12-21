#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("/content/drive/MyDrive/dataset/discritiexefadfas.csv")
data['class'] = (data['class']== ' >50K.')*1
y = data[['class']].copy()
x= data.drop('class', axis=1)

# print(data1.head())
#print(x.head())
#print(y.head(),'Hi')

for label in x.columns:
  x[label]= LabelEncoder().fit_transform(x[label])
# use an encoder
x_train, x_test_valid, y_train, y_test_valid= train_test_split(x,y,test_size= 0.5,random_state=0)
x_val, x_test, y_val, y_test= train_test_split(x_test_valid,y_test_valid,test_size= 0.5,random_state=0)
# print('head of val')


# print(x_train.head())
# print(y_train.head(),'Ok')



# Train your decision tree model on the training data
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

#training accuracy
y_tr_pred = classifier.predict(x_train)
tr_accuracy = accuracy_score(y_train, y_tr_pred)
print(tr_accuracy)

# Evaluate the model's performance on the validation set
y_val_pred = classifier.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(val_accuracy)



x_ax=[]
y_ax_test=[]
y_ax_train=[]
y_ax_valid=[]
min=1
for n in range(1,33,1):
  model = DecisionTreeClassifier(max_depth=n, criterion='entropy', random_state=0)
  model.fit(x_train,y_train)
  x_ax.append(n)
  y_ax_test.append(model.score(x_test,y_test))
  y_ax_train.append(model.score(x_train,y_train))
  y_ax_valid.append(model.score(x_val,y_val))

plt.plot(x_ax,y_ax_train)
plt.plot(x_ax,y_ax_valid)
plt.plot(x_ax,y_ax_test)
plt.legend(['Training data','Validation data','Testing data'])
plt.xlabel("Depth of Decision Tree")
plt.ylabel("Accuracy")


#prune to 5 nodes
classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(x_train, y_train)

#training accuracy
y_tr_pred = classifier.predict(x_train)
tr_accuracy = accuracy_score(y_train, y_tr_pred)
print(tr_accuracy)

# Evaluate the model's performance on the validation set
y_val_pred = classifier.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(val_accuracy)
