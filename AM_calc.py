import pandas as pd


# 将B类报表的数据删除
def delete_b(df):
    df = df[df['Typrep'].isin(['A'])]
    df.to_excel("D:\\study\\Quantitative_investment\\table2.xlsx", index=False)
    return df


def calc_AM(value, Msmvosd):
    AM = value / Msmvosd
    AM = pd.DataFrame(AM)
    AM.to_csv('D:\\study\\Quantitative_investment\\AM.csv', index=False)
    return AM


# 剔除B类报表的信息
df = pd.read_excel('D:\\study\\Quantitative_investment\\table.xlsx')
df = delete_b(df)

# 检验缺失值
value = df.iloc[:, 3]
num_nan = value.isnull().sum()
print(num_nan)

# 计算AM因子
Msmvosd = df.iloc[:, 4]
calc_AM(value, Msmvosd)
