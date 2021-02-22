from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import plot
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_csv('diabetes_data_upload.csv', header=0)
print (df.head())

le = preprocessing.LabelEncoder()
for column in df:
    df[column] = le.fit_transform(df[column])

Y = df[['class']].copy()
X = df.drop('class', axis=1)

trainX, testX, trainY, testY = train_test_split(X, Y)
clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion="entropy"), n_estimators=100, random_state=0)
clf = clf.fit(trainX, trainY)
#plt.figure(figsize=(14,14))
#tree.plot_tree(clf, fontsize=10, feature_names = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
#                                                  'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
#                                                 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
#                                                 'muscle stiffness', 'Alopecia', 'Obesity'], class_names=['0', '1'])
#plt.show()
predY = clf.predict(testX);


print("train error without pruning: " + str(1-clf.score(trainX, trainY)))
print("test error without pruning: " + str(1-clf.score(testX, testY)))
#print("tree depth without pruning : " + str(clf.tree_.max_depth))

scores = cross_val_score(clf, X, Y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f when used cross validation without pruning" % (scores.mean(), scores.std()))

title = "No Pruning Training Performance"
plot.plot_learning_curve(clf, title, trainX, trainY)
plt.show()

title = "No Pruning Test Performance"
plot.plot_testing_curve(clf, title, testX, testY)
plt.show()

#df.Age.hist()
#plt.show()

#df.Polyuria.hist()
#plt.show()

#df.Polydipsia.hist()
#plt.show()


