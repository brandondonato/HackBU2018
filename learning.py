import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")
import pandas as pd


#Actually using our data now
def BuildDataSet(features=["OPEN", "CLOSE"]):
	data_df = pd.read_csv("coinmarketbtc.csv")
	X = np.array(data_df[features].values)#.tolist())
	y = (data_df["VALUE"].values.tolist())
	return X,y

def Analysis():
	X, y = BuildDataSet()
	clf = svm.SVC(kernel = 'linear', C=1.0)
	clf.fit(X,y)

	w = clf.coef_[0]
	a = -w[0]/w[1]
	xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
	yy = a * xx - clf.intercept_[0]/w[1]

	h0 = plt.plot(xx,yy, 'k-', label="non weighted")

	plt.scatter(X[:, 0], X[:, 1])
	plt.legend()
	plt.show()

Analysis()





'''
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, .6, 11]

X = np.array([[1,2],
 [5,8],
 [1.5,1.8],
 [8,8],
 [1,.6],
 [9,11],])
#Grouping them by large or smaller numbers
y = [0,1,0,1,0,1]
#Classifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X,y)

print(clf.predict([[0.58,0.76]]))

w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(0, 12)
yy = a * xx - clf.intercept_[0]/w[1]
h0 = plt.plot(xx,yy, '-k', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend( )
plt.show()
'''