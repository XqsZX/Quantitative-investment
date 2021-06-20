import pandas as pd
import statistics
import numpy as np


# 将B类报表的数据删除
def delete_b(df):
    df = df[df['Typrep'].isin(['A'])]
    df.to_excel("D:\\study\\Quantitative_investment\\ZCZJ.xlsx", index=False)
    return df


# 计算统计值
def calc(list):
    # 计算算数平均数
    mean = statistics.mean(list)
    # 方差
    var = statistics.pvariance(list)
    # 最大最小值
    mini = min(list)
    maxi = max(list)
    # 中位数，众数
    med = statistics.median(list)
    multi = statistics.mode(list)
    # 样本数，缺失值
    num = len(list)
    num_nan = list.isnull().sum()
    # 四个四分位数
    Quartile_25 = np.percentile(list, 25)
    Quartile_50 = np.median(list)
    Quartile_75 = np.percentile(list, 75)
    print("平均数为：{}".format(mean))
    print("方差数为：{}".format(var))
    print("最大最小值为：{}；{}".format(mini, maxi))
    print("中位数和众数为：{}；{}".format(med, multi))
    print("样本数和缺失值为：{}；{}".format(num, num_nan))
    print("25分位数，50分位数，75分位数为：{}；{}；{}".format(Quartile_25, Quartile_50, Quartile_75))


# sheet = pd.read_excel('D:\\study\\Quantitative_investment\\test.xlsx')
df = pd.read_excel("D:\\study\\Quantitative_investment\\数据收集\\FS_Combas.xlsx")
# 删除B类资产负债表
df = delete_b(df)
value = df.iloc[:, 3]
calc(value)

