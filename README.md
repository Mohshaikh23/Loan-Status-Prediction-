# Loan-Status-Prediction-
Model Building using SVM for Loan Prediction using Application dataset
# LOAN STATUS PREDICTION

### Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Importing data into notebook

df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")
df.head()

### Evaluating the Dataset

type(df)

df.shape

df.describe()

df.info()

## EXPLORATORY DATA ANALYSIS

### Finding Missing Values

df.isna().sum()

### Dropping the missing Values

df = df.dropna()

df.isna().sum()

### Converting Target Feature to numeric in nature

#The Loan Status is attributed into Categorical column, so we need to convert for further modelling.
df.replace({"Loan_Status":{"N":0, "Y":1}}, inplace = True)

df.head()

df['Dependents'].value_counts()

#The 3+ number will present errors while handling into model, so conversion is required
df = df.replace(to_replace = '3+', value = 4)

df['Dependents'].value_counts()

### Visualuizing the Segments for Education 

sns.countplot(x = "Education", hue = "Loan_Status", data = df)
plt.show()

### Visualuizing the Segments for Marriage Segment 

sns.countplot(x = "Married", hue = "Loan_Status", data = df)
plt.show()

#### Removing non required Feature

df.drop("Loan_ID", axis = 1, inplace = True)

df

#### The dataset requires to be submerged into numerical Values

df = pd.get_dummies(df, drop_first = True)

df

#### Splitting data into X And Y features

X = df.drop("Loan_Status", axis = 1)
y =df["Loan_Status"]

X

y

#Importing train_test_split to the dataset for implementing into the model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 55)

X_train.head()

X_test.head()

y_train.head()

y_test.head()

X.shape

X_train.shape

X_test.shape

### Importing SVM Library

from sklearn import svm

#assigning the SVM Classifier to a variable
classifier = svm.SVC(kernel='linear')

classifier

#Fitting the data to the classifier model
classifier.fit(X_train, y_train)

#For Checking how accuarate the model is, we need to import accuracy score
from sklearn.metrics import accuracy_score

#predicting the value for the required Target Variable 
X_train_prediction = classifier.predict(X_train)

#Calculating the Accuracy for the model
training_data_accuracy  =  accuracy_score(X_train_prediction, y_train)

print (" the Accuaracy for the Training Dataset is ",round(training_data_accuracy*100,2),"%")

#Similarly applying the testing dataset to the model 
X_test_prediction = classifier.predict(X_test)
test_data_accuracy  =  accuracy_score(X_test_prediction, y_test)

print (" the Accuaracy for the Test Dataset is ",round(test_data_accuracy*100,2),"%")

### Thus the Accuracy for the model performed here is predicting 81.25% Accurately.

