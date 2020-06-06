import csv
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import set_option
import sys
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics._classification import classification_report, confusion_matrix
from sklearn.svm._classes import SVR
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
filename = "./combo.csv"
total_list = [210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
              310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0,
              410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
              510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0,
              610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0,
              710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0,
              810.0, 820.0, 830.0, 840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0,
              910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0, 'classification']
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',')
dataset = read_csv(filename, names=total_list)

print(dataset.groupby('classification').size())
print(dataset.shape)
array = dataset.values
X = array[:, 0:80]
Y = array[:, 80]
validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                                                test_size=validation_size, random_state=seed)

# Spot-Check Algorithms
models = []

models.append(('KNN7', KNeighborsClassifier(n_neighbors=7)))
models.append(('KNN9', KNeighborsClassifier(n_neighbors=9)))
models.append(('KNN11', KNeighborsClassifier(n_neighbors=11)))
models.append(('DT', DecisionTreeClassifier()))

# results = []
# names = []
# # for name, model in models:
# #     kfold = KFold(n_splits=10, shuffle=True)
# #     cv_results = cross_val_score(
# #         model, X_train, Y_train, cv=kfold, scoring='accuracy')
# #     cv_results = cv_results
# #     results.append(cv_results)
# #     names.append(name)
# #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# #     print(msg)

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# kneighbors
# alg = KNeighborsClassifier(n_neighbors=7)
# alg.fit(X_train, Y_train)
# predictions = alg.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions, zero_division=1))

# alg = KNeighborsClassifier(n_neighbors=9)
# alg.fit(X_train, Y_train)
# predictions = alg.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions, zero_division=1))

# alg = KNeighborsClassifier(n_neighbors=11)
# alg.fit(X_train, Y_train)
# predictions = alg.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions, zero_division=1))

# decision tree classifier
alg = DecisionTreeClassifier()
alg.fit(X_train, Y_train)
predictions = alg.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions, zero_division=1))
