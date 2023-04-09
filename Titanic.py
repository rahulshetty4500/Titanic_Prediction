#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# Load the dataset into your program
data = pd.read_csv('/Users/rahul/Downloads/titanic (1).csv')

def normalize_z(data):
    dfout = (data - data.mean(axis=0))/data.std(axis=0)
    return dfout

x=normalize_z(X)

X_train=X[:150]
X_test=X[150:]
y_train=y[:150]
y_test=y[150:]
X_train

# Remove any missing or irrelevant data
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data = data.dropna()

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Split the data into training and testing sets using a specified ratio
X = data.drop(['Survived'], axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features using standardization
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# Train and evaluate the logistic regression model
lr = LogisticRegression(lr=0.1, n_iters=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('Logistic Regression accuracy:', accuracy_score(y_test, y_pred))
print('Logistic Regression precision:', precision_score(y_test, y_pred))
print('Logistic Regression recall:', recall_score(y_test, y_pred))
print('Logistic Regression F1-score:', f1_score(y_test, y_pred))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[ ]:




