from sklearn import tree
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import plot
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_csv('MI.csv', header=0)
print (df.head())

le = preprocessing.LabelEncoder()
for column in df:
    df[column] = le.fit_transform(df[column])

Y = df[['FIBR_PREDS','PREDS_TAH','JELUD_TAH','FIBR_JELUD','A_V_BLOK','OTEK_LANC','RAZRIV','DRESSLER','ZSN','REC_IM','P_IM_STEN','LET_IS']].copy()
X = df.drop(['FIBR_PREDS','PREDS_TAH','JELUD_TAH','FIBR_JELUD','A_V_BLOK','OTEK_LANC','RAZRIV','DRESSLER','ZSN','REC_IM','P_IM_STEN','LET_IS'], axis=1)

trainX, testX, trainY, testY = train_test_split(X, Y)
clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion="entropy"), n_estimators=100, random_state=0)
start = time.time()

clf = clf.fit(trainX, trainY['FIBR_PREDS'])
end = time.time()
print ("Training speed With No Pruning(seconds): " + str(end-start))
predTrainY = clf.predict(trainX)

start = time.time()
predY = clf.predict(testX)
end = time.time()
print ("Testing speed with No Pruning(seconds) : " + str(end-start))

print("")
for column in trainY:
    print("train score for " + column + " :" + str(1-accuracy_score(predTrainY, trainY[column])))
    print("test score for " + column + " :" + str(1-accuracy_score(predY, testY[column])))

scores = cross_val_score(clf, X, Y['FIBR_PREDS'], cv=5)
print("%0.2f error with a standard deviation of %0.2f when used cross validation with pruning" % (1-scores.mean(), scores.std()))
title = "No Pruning Training Performance"
plot.plot_learning_curve(clf, title, trainX, trainY['FIBR_PREDS'])
plt.show()

title = "No Pruning Test Performance"
plot.plot_testing_curve(clf, title, trainX, trainY['FIBR_PREDS'])
plt.show()
