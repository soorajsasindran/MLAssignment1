from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import plot
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('Skin_NonSkin.txt',sep='\s+',header=None)
df.columns =['B', 'G', 'R', 'Skin']
print (df.head())


le = preprocessing.LabelEncoder()
for column in df:
    df[column] = le.fit_transform(df[column])

Y = df[['Skin']].copy()
X = df.drop('Skin', axis=1)

trainX, testX, trainY, testY = train_test_split(X, Y)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (15, ), random_state = 1)
print("start")
clf = clf.fit(trainX, trainY)
print("done")
predY = clf.predict(testX)
scores = cross_val_score(clf, X, Y, cv=5)

print("train error without pruning: " + str(1-clf.score(trainX, trainY)))
print("test error without pruning: " + str(1-clf.score(testX, testY)))
print("%0.2f error when used cross validation without pruning" % (1-scores.mean()))

title = "No Pruning Training Performance"
plot.plot_learning_curve(clf, title, trainX, trainY)
plt.show()

title = "No Pruning Test Performance"
plot.plot_testing_curve(clf, title, testX, testY)
plt.show()


