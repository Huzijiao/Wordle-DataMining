import math

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, plot_importance
import warnings
matplotlib.use("TkAgg")

# gain data
df = pd.read_excel('ProblemC.xlsx')
X = df[['k','P_s','Pos_Percentage']]
Y = df[['Mean', 'Var']]
df1 = pd.read_excel('Test.xlsx')
X1 = df1[['k','P_s','Pos_Percentage']]
# print(X)
# for j in range(40,100):
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
Shu1 = []
Shu2 = []
Shu3 = [1,2,3,4,5]
    # parameters
for i in range(16,17):
    other_params = {'learning_rate': i*0.001, 'n_estimators': 300, 'max_depth':4, 'min_child_weight': 1, 'seed': 0,
                        'subsample': 1, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,}
    multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params)).fit(train_X,train_y)
    # kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    # kf = KFold(n_splits=5,shuffle=True)
    # X = np.array(X)
    # Y = np.array(Y)
    # for train_Index,test_Index in kf.split(X):
    #     train_X = X[train_Index]
    #     test_X = X[test_Index]
    #     train_Y = Y[train_Index]
    #     test_Y = Y[test_Index]
    #     model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params)).fit(train_X,train_Y)
    #     result = model.predict(test_X)
    #     Sum = 0
    #     Sum1 = 0
    #     for i in range(len(result)):
    #         print(str(test_Y[i])+" "+str(result[i]))
    #         Mean1 = float(test_Y[i][0])
    #         Mean2 = float(result[i][0])
    #         Var1 = float(test_Y[i][1])
    #         Var2 = float(result[i][1])
    #         w1 = abs(Mean2-Mean1)/abs(Mean1)
    #         w2 = abs(Var2-Var1)/abs(Var1)
    #         Sum += w1*w1
    #         Sum1 += w2*w2
    #     Shu1.append(math.sqrt(Sum/(len(result)-1)))
    #     Shu2.append(math.sqrt(Sum1/(len(result)-1)))
    print(multioutputregressor.predict(X1))
    # check = multioutputregressor.predict(test_X)
    # print(check)
    # data2 = test_y.iloc[:,0:2].values
    # max1 = 0
    # max2 = 0
    # w1 = 0
    # w2 = 0
    # for j in range(0,len(check)):
    #         a = check[j]
    #         b = data2[j]
    #         a1 = float(a[0])
    #         a2 = float(a[1])
    #         b1 = float(b[0])
    #         b2 = float(b[1])
    #         wu1 = abs(b1-a1)/a1
    #         wu2 = abs(b2-a2)/a2
    #         w1 += wu1*wu1
    #         w2 += wu2*wu2
    #         max1 = max(max1,abs(b1-a1)/a1)
    #         max2 = max(max2,abs(a2-b2)/a2)
    # Shu1.append(math.sqrt(w1/(len(check)-1)))
    # Shu2.append(math.sqrt(w2/(len(check)-1)))
    # Shu3.append(i)
        # u = [['k','P_s','Pos_Percentage'],[3,4,4.505024614988391]]
    # check = multioutputregressor.predict(X1)
    # print(check)
    # dataFrame = pd.DataFrame({'a':Shu1,'b':Shu2,'c':Shu3})
    # dataFrame.to_csv("Shuju.csv")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# fig =  plt.figure(figsize=(7.5, 5.8))
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#
# x = np.array(Shu3)
# y = np.array(Shu1)
# y1 = np.array(Shu2)
# axes.plot(x, y, c="green", label=r'$Mean$', ls='-.', alpha=0.6, lw=2, zorder=2)
# axes.plot(x, y1, c="blue", label=r'$Var$', ls=':', alpha=1, lw=1, zorder=1)
# plt.xlabel("5-Fold Cross-Validation")
# plt.ylabel("Standard deviation of relative error")
# axes.legend()
# plt.show()
