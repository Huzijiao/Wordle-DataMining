import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import csv
import math

def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)

# dataframe = pd.read_csv('BCHAIN-MKPRU.csv', usecols=[1], engine='python')
data = pd.read_excel('./data/Problem_C_Data_Wordle.xlsx') # 默认读取第一个sheet
dataset = numpy.array(data.iloc[1:,[4]].values.tolist()[::-1]) #读取指定列的所有行数据

# print(dataset)
print("请输入一个大于等于50的整数:")
k = eval(input())

while(k<50):
    print("不理想，请重新输入")
    k = eval(input())
if k>=len(dataset):
    k = len(dataset)
dataset = dataset[:k]

# 将整型变为float
dataset = dataset.astype('float32')
#归一化 在下一步会讲解
#上面代码的片段讲解
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.65)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

#训练数据太少 look_back并不能过大
if len(dataset)<50:
    print("注意效果可能不佳！")
    look_back = 1
elif len(dataset)<200:
    look_back = 2
elif len(dataset)<500:
    look_back = 3
elif len(dataset)<1000:
    look_back = 4
else:
    look_back = 5

trainX,trainY  = create_dataset(trainlist,look_back)
testX,testY = create_dataset(testlist,look_back)
print(trainX.shape)
print(trainY.shape)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=90, batch_size=1, verbose=2)
model.save(os.path.join("DATA","Test" + ".h5"))
# make predictions

#model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(len(dataset))
Predict = model.predict(dataset.reshape(-1,1,1))

#反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
Predict = scaler.inverse_transform(Predict)
dataset = scaler.inverse_transform(dataset)
# 反归一化调整
Predict *= (dataset[-1][0]/Predict[-2][0])
for i in range(len(Predict)-1):
    Predict[i] *= math.e ** (0.0054 * ((len(Predict)-2)-i))
# Predict += Predict * MAPE / 100

# plt.plot(trainY)
# plt.plot(trainPredict[1:])
# plt.show()
# plt.plot(testY)
# plt.plot(testPredict[1:])
# plt.show()
# print(type(Predict))
plt.figure(figsize=(16,9))
plt.plot(dataset,label='data')
# .values.tolist().insert(0,dataset[0])
plt.plot(numpy.insert(Predict, 0, dataset[0]),label='predict')
# plt.title('Bit_coin')
plt.xlabel("Date")
plt.ylabel("Number_of_reported_results")
plt.legend()
plt.grid()  # 添加网格
plt.show()
# plt.savefig("预测.png")
print(len(dataset))
print(len(Predict))
# print(dataset[-1] / Predict[-2])
# print(Predict)

print("预测值为：")
print(Predict[-1][0])

with open('Gold_p01.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(6,len(Predict)):
        if Predict[i][0] >= dataset[i-1][0]:
            writer.writerow([1])
        else:
            writer.writerow([0])

# 计算误差
i=0
MAPE=0
for yi in dataset:
    MAPE += abs((Predict[i+1][0]-yi[0])/yi[0])
    if i==len(dataset)-2:
        break
    i+=1
MAPE = MAPE * 100 / len(dataset)
print("平均绝对百分比误差MAPE（Mean Absolute Percentage Error）:")
print(MAPE,"%")

i=0
RMSE=0
for yi in dataset:
    RMSE += (Predict[i+1][0]-yi[0])**2
    if i==len(dataset)-2:
        break
    i+=1
print("均方误差MSE（Mean Square Error）")
print(RMSE)
RMSE = (RMSE / len(dataset))**0.5
print("均方根误差RMSE（Root Mean Square Error）:")
print(RMSE)

i=0
MAE=0
for yi in dataset:
    MAE += abs((Predict[i+1][0]-yi[0]))
    if i==len(dataset)-2:
        break
    i+=1
MAE = MAE / len(dataset)
print("平均绝对误差（Mean Absolute Error）:")
print(MAE)


DS=0
for i in range(1,len(dataset)-1):
    if (Predict[i+1][0]-Predict[i][0])*(dataset[i+1][0]-dataset[i][0])>0:
        DS += 1
DS = DS / (len(dataset)-1)
print("方向对称性（Directional Symmetry）:")
print(DS)