import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pandas import DataFrame
from numpy import zeros, empty
from sklearn.linear_model import LinearRegression


path = 'D:\\study\\Quantitative_investment\\bond.xlsx'
data = pd.read_excel(path, index_col=False, dtype={"t": float, "y": float})
t = data["t"]
y = data["y"]
j = 0.5
i = 0
A = []
B = []
C = []
D = []
E = []
F = []
a = zeros((len(t), 19))
b = zeros((len(t), 19))
while j < 5.5:
    for m in range(len(t)):
        a[m][i] = (1 - math.exp(-t[m] / j)) / (t[m] / j)
        b[m][i] = (1 - math.exp(-t[m] / j)) / (t[m] / j) - math.exp(-t[m]/ j)
    model = smf.ols(formula='y ~ a[:,i] + b[:,i]', data=data).fit()
    A.append(model.params.values)
    B.append(model.rsquared)
    C.append(model.fvalue)
    D.append(model.f_pvalue)
    E.append(model.mse_model)
    F.append(model.tvalues.values)
    j += 0.5
    i += 1
j = 6

while 5.5 < j < 11:
    for m in range(len(t)):
        a[m][i] = (1 - math.exp(-t[m] / j)) / (t[m] / j)
        b[m][i] = (1 - math.exp(-t[m] / j)) / (t[m] / j) - math.exp(-t[m]/ j)
    model = smf.ols(formula='y ~ a[:,i] + b[:,i]', data=data).fit()
    A.append(model.params.values)
    B.append(model.rsquared)
    C.append(model.fvalue)
    D.append(model.f_pvalue)
    E.append(model.mse_model)
    F.append(model.tvalues.values)
    j+=1
    i+=1
j=15
while 11<j<35:
    for m in range(len(t)):
        a[m][i] = (1 - math.exp(-t[m] / j)) / (t[m] / j)
        b[m][i] = (1 - math.exp(-t[m] / j)) / (t[m] / j) - math.exp(-t[m] / j)
    model = smf.ols(formula='y ~ a[:,i] + b[:,i]', data=data).fit()
    A.append(model.params.values)
    B.append(model.rsquared)
    C.append(model.fvalue)
    D.append(model.f_pvalue)
    E.append(model.mse_model)
    F.append(model.tvalues.values)
    j += 5
    i += 1

b0 = zeros((19, 1))
b1 = zeros((19, 1))
b2 = zeros((19, 1))
for i in range(19):
    b0[i][0]=A[i][0]
    b1[i][0]=A[i][1]
    b2[i][0]=A[i][2]
#print(b2)

shu = []
all = []

shu.append(B)
shu.append(C)
shu.append(D)
shu.append(E)
all = np.transpose(shu)
# all.append(A)
# all.append(F)

all = pd.DataFrame(all)
A = pd.DataFrame(A)
F = pd.DataFrame(F)

# print(all)
# print(A)
# print(F)
result = pd.concat([A, F, all], axis=1)
result.to_excel("D:\\study\\Quantitative_investment\\result.xlsx", index=False)

# print(A)
