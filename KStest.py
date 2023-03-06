import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_excel("Problem_C_Data_Wordle_pre.xlsx")
df1 = df[['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)']]
df1 = np.array(df1)
pvalue  = 1
for i in range(0,len(df1)):
    data = df1[i]
    data = np.array(data)
    Mean = data.mean()
    Std = data.std()
    a = stats.kstest(data, 'norm', (Mean, Std))
    a = list(a)
    pvalue = min(pvalue,float(a[1]))
print(pvalue)
