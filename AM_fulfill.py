import pandas as pd
import matplotlib.pyplot as plt


def fulfill(df):
    # 求列长度
    length = len(df.iloc[:, 0])
    # print(length)
    # 求股票数量
    num_of_stock = len(df.iloc[0, :]) - 1
    # print(num_of_stock)
    # 定义tem数组存储最后一个非空元素的索引
    tem = [0] * num_of_stock
    # 找出最后一个非空AM值的索引
    for i in range(num_of_stock):
        tem[i] = df.iloc[:, i + 1].last_valid_index()
    # 将tem数组中None值（一整列都是空的）替换为276
    for j in range(num_of_stock):
        if tem[j] is None:
            tem[j] = 276
    # 将所有退市后的AM值设为inf
    for k in range(num_of_stock):
        df.iloc[tem[k] + 1:length, k + 1] = 'inf'
    # 用前面最近的已知值对nan值进行填充
    df = df.fillna(axis=0, method='ffill')
    # 最后把之前设为inf值的地方重新修改为nan
    for m in range(num_of_stock):
        df.iloc[tem[m] + 1:length, m + 1] = 'NA'
    # 填充完后写入文件
    df.to_excel('D:\\study\\Quantitative_investment\\数据收集\\AM_fulfill.xlsx', index=False)


def plot(df):
    # 读取万达数据部分
    wanda = df.iloc[:, 2]
    time = df.iloc[:, 0]

    # 绘图部分
    plt.figure(figsize=(8, 4.5), dpi=100)
    plt.xlabel('time', color='black', fontsize=15)  # x标签
    plt.ylabel('wanda', color='black', fontsize=15)  # y标签
    plt.plot(time, wanda)
    plt.title('sequence chart', size=20)
    plt.savefig("wanda.png", dpi=100)
    plt.show()


# 读取文件
df = pd.read_excel('D:\\study\\Quantitative_investment\\数据收集\\table3.xlsx')
fulfill(df)
wanda = pd.read_excel('D:\\study\\Quantitative_investment\\数据收集\\AM_fulfill.xlsx')
plot(wanda)
