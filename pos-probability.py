import pandas as pd
import operator
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# f = open("words.txt", encoding="utf-8")
str = ""
ban = ["-", "\'", "/", "3", "2", ".", "0", "1", ","]
with open("words.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        flag = 0
        for zifu in ban:
            if zifu in line:
                flag = 1
        if flag:
            continue
        if len(line) == 5:
            str += line.lower()
dict = {}
# 带位置的字典
posdict = [{}, {}, {}, {}, {}]
pos = 0
for i in str:
    dict[i] = dict.get(i, 0) + 1
    posdict[pos % 5][i] = posdict[pos % 5].get(i,0) + 1
    pos += 1

# sorted 方法会生成一个排序好的容器
# operator.itemgetter(1)  获取字典第一维的数据进行排序
# reverse 表示倒序排列
dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
print(dict)

print(posdict)
zimu = []
shuzi = []
for cipin in dict:
    zimu.append(cipin[0])
    shuzi.append(cipin[1])
# zimu.pop()
# shuzi.pop()
# print(zimu[-1])
#数据可视化
sns.barplot(x=zimu,y=shuzi)
plt.xlabel("Letter", fontsize=10)
plt.ylabel("Frequency", fontsize=14)
plt.show()


