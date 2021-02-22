from sklearn import tree
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import plot
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score


df = pd.read_csv('MI.csv', header=0)
print (df.head())

le = preprocessing.LabelEncoder()
for column in df:
    df[column] = le.fit_transform(df[column])

Y = df[['FIBR_PREDS','PREDS_TAH','JELUD_TAH','FIBR_JELUD','A_V_BLOK','OTEK_LANC','RAZRIV','DRESSLER','ZSN','REC_IM','P_IM_STEN','LET_IS']].copy()
X = df.drop(['FIBR_PREDS','PREDS_TAH','JELUD_TAH','FIBR_JELUD','A_V_BLOK','OTEK_LANC','RAZRIV','DRESSLER','ZSN','REC_IM','P_IM_STEN','LET_IS'], axis=1)

trainX, testX, trainY, testY = train_test_split(X, Y)
clf = tree.DecisionTreeClassifier(criterion="entropy")
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

print("train score for " + 'FIBR_PREDS' + " :" + str(accuracy_score(predTrainY, trainY['FIBR_PREDS'])))
print("test score for " + 'FIBR_PREDS' + " :" + str(accuracy_score(predY, testY['FIBR_PREDS'])))



start = time.time()
predY = clf.predict(testX)
end = time.time()
print ("Testing speed with No Pruning(seconds) : " + str(end-start))

title = "No Pruning Training Performance"
plot.plot_learning_curve(clf, title, trainX, trainY['FIBR_PREDS'])
plt.show()

title = "No Pruning Test Performance"
plot.plot_testing_curve(clf, title, trainX, trainY['FIBR_PREDS'])

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=clf.tree_.max_depth-2)
start = time.time()
clf = clf.fit(trainX, trainY['FIBR_PREDS'])
end = time.time()
print ("Training speed With  Pruning(seconds): " + str(end-start))
predTrainY = clf.predict(trainX)

print("")

print("train score for " + 'FIBR_PREDS' + " :" + str(1-accuracy_score(predTrainY, trainY['FIBR_PREDS'])))
print("test score for " + 'FIBR_PREDS' + " :" + str(1-accuracy_score(predY, testY['FIBR_PREDS'])))
scores = cross_val_score(clf, X, Y['FIBR_PREDS'], cv=5)
print(scores)
print("%0.2f error with a standard deviation of %0.2f when used cross validation with pruning" % (1-scores.mean(), scores.std()))

title = "Pruning Training Performance"
plot.plot_learning_curve(clf, title, trainX, trainY['FIBR_PREDS'])
plt.show()

title = "Pruning Test Performance"
plot.plot_testing_curve(clf, title, trainX, trainY['FIBR_PREDS'])
plt.show()