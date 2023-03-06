import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.utils import column_or_1d

df = pd.read_excel("DBSCAN_new.xlsx")
X = df[['try_mean','try_var']]
Y = df[['cluster_db']]
Y = column_or_1d(Y, warn=True)
train_X,test_X,train_Y,test_Y = train_test_split(X,Y, test_size=0.2, random_state=102)
clf = MLPClassifier(activation='logistic',solver='lbfgs',max_iter = 500,alpha = 1e-5,hidden_layer_sizes = (100,50),random_state = 1,verbose = False)
clf.fit(train_X,train_Y.ravel())
Result = clf.predict(test_X)
# print(Result)
Sum = 0
Sum1 = 0
for i in range(0,len(test_Y)):
    if(str(test_Y[i])=="-1"):
        Sum += 1
        if(str(Result[i])=="-1"):
            Sum1 += 1
if (Sum == 0):
    print("0 1.0")
else:
    print(str(Sum) + " " + str(Sum1 / Sum))

print(clf.predict(np.array([[4.40,2.24]])))
    # print(clf.score(test_X,test_Y))

