from __future__ import division, print_function
import os
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error


data_path = os.path.join(os.getcwd(), 'data', 'regression_part1.csv')
df = pd.read_csv(data_path, delimiter = ',')

# print(df.shape)
# print(df.describe())

x = df['revision_time'].values.reshape(-1,1)
xtrain = np.insert(x,0,1,axis=1)

ytrain = df['exam_score'].values.reshape(-1,1)
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=1000)

lm = LinearRegression(fit_intercept=False)
reg = lm.fit(xtrain, ytrain)
print(reg.coef_)
# print(lm.score(xtrain, ytrain))
# print(lm.score(xtest, ytest))
# print(reg.predict(xtrain))

b = inv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(ytrain)
# print(b)
print(b)
yhat = xtrain.dot(b)
print(yhat.shape)
def fit_scatter(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

fit_scatter(ytrain, reg.predict(xtrain))
plt.scatter(x, ytrain, color='k')
plt.scatter(x, reg.predict(xtrain), color='g')
plt.plot(x, reg.predict(xtrain), 'r--', lw = 2)
plt.title('Revision time vs Exam score')
plt.xlabel('Revision Time')
plt.ylabel('Exam Score')
plt.show()

print(mean_squared_error(ytrain, yhat))
print(mean_squared_error(ytrain, reg.predict(xtrain)))
