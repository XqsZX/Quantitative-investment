import pandas as pd
import numpy as np
# 读取excel文档
df = pd.read_excel('D:\\study\\Quantitative_investment\\return2.xlsx')
# print(len(df.loc[1]))
# 设置一个空的二维数组
mom6 = [[] for i in range(len(df.loc[1]))]
# 设置一个计数器
count = 0
for i in range(len(df.loc[1]) - 1):
    # 读不同股票的数据
    r = df.iloc[:, i + 1]
    # print(r)
    for j in range(len(df.iloc[:, 2]) - 6):
        # 将计算的值写入二维数组中，如果其中有nan值，那么直接返回nan
        if r[j: j + 6].isna().sum() >= 2:
            mom6[count].append(np.nan)
        else:
            mom6[count].append(np.nansum(r[j: j + 6]))
    # 计数器加一
    count = count + 1
    print(count)

# 将二维数组转换成dataframe格式，准备将其写入excel中
mom6 = pd.DataFrame(mom6)
# 将二维数组行列互换，以适应数据格式
mom6_T = mom6.T
# 看看长度，确定已经行列互换
print(len(mom6_T))
# 进行数据填充
mom6_final = mom6_T.fillna(axis=0, method='ffill')
# 将得到的数组写入文件
mom6_final.to_csv('D:\\study\\Quantitative_investment\\mom6.csv', index=False)
