import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from NWttest import nwttest_1samp
from scipy import stats
import statsmodels.api as sm


# 根据输入的行数，对不同时间的AM值进行排序，并返回排序后的股票代码
def stock_codes(df, col):
    # 筛选数据
    AM = df.iloc[:, 1:]
    # 让数据按照降序的方式进行排列
    AM.sort_values(by=[col], axis=1, na_position='first', inplace=True, ascending=False)
    count = 0
    # 剔除所有值为nan的列
    num_of_nan = AM.iloc[col, :].isna().sum()
    AM = AM.iloc[:, num_of_nan:]
    # 写入文件（也可以不写）
    # AM.to_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\AM_201812.xlsx', index=False)
    # 输出股票代码（根据据AM值排序）
    codes = AM.iloc[:1, :]
    return codes


# 打印出AM前十的股票收益率
def find_r(list):
    # 读取含有收益率的文件
    df = pd.read_csv('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\predictions_1.csv')
    # df.fillna(value=0, inplace=True)
    # 对时间进行筛选
    df_201812 = df.iloc[251, :]
    # print(df_201812)
    # 对应输出
    for i in list:
        print("代号为{}的股票在2018.12股票预期收益率为{}".format(i, df_201812[i]))


# 输入股票代码（codes）
def calc(codes, col):
    # 读取含有收益率的文件
    df = pd.read_csv('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\final_return1.csv')
    df.fillna(value=0, inplace=True)
    # 对时间进行筛选
    df_r = df.iloc[col, :]
    # 计算每一组的股票数量
    group = math.ceil(len(codes) / 10)
    # 第一组做空的股票数量
    empty = len(codes) - 9 * group
    # 定义list，用来存储收益率
    r = [0] * 10
    # 先计算第一组
    for i in codes[0: empty]:
        r[0] = r[0] + df_r[i]
    # 计算第一组的收益率
    r[0] = r[0] / empty
    # 计算之后几组
    for j in range(1, 10):
        for k in codes[empty + group * (j - 1): empty + group * j]:
            r[j] = r[j] + df_r[k]
        # 计算等权平均收益
        r[j] = r[j] / group
    print(r)
    # 返回投资组合收益和多空组合收益
    return r, r[9] - r[0]


# 计算时间序列收益
def time_series(df):
    # 定义一个276 * 10的数组存储投资组合收益率
    ret = np.zeros((252, 10))
    # 定义数组存储多空组合收益率
    duokong = [0] * 252
    # 根据上面两个函数得到股票代码序列，投资组合收益和多空组合收益
    for i in range(252):
        AM = np.array(stock_codes(df, i).columns)
        AM = AM.tolist()
        ret[i], duokong[i] = calc(AM, i + 12)

    # 将得到的数据转化为dataframe格式，以便之后写入excel文件中
    ret = pd.DataFrame(ret)
    duokong = pd.DataFrame(duokong)
    # 将结果写入excel中
    ret.to_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\ret.xlsx')
    duokong.to_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\duokong.xlsx')


# 绘制时序收益图
def plot_time_series():
    ret = pd.read_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\ret.xlsx')
    # 读取时间序列数据部分
    time = ret.iloc[:, 0]

    # 绘图部分
    plt.figure(figsize=(8, 4.5), dpi=100)
    # 定义坐标轴名
    plt.xlabel('time', color='black', fontsize=15)  # x标签
    plt.ylabel('return', color='black', fontsize=15)  # y标签
    # 绘制10条曲线
    for i in range(1, 11):
        plt.plot(time, ret.iloc[:, i])
    # 增加图例
    plt.legend(['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
                'eighth', 'ninth', 'tenth'], loc='upper right', ncol=2)
    # 增加标题
    plt.title('time series chart', size=20)
    # 对图片进行保存
    plt.savefig("time series.png", dpi=100)
    plt.show()


# CAPM估计α
def CAPM(ret, rf, risk_premium):
    # 添加常数列
    X = sm.add_constant(risk_premium)
    # 进行OLS拟合
    model = sm.OLS(ret - rf, X)
    # 拟合结果
    results = model.fit()
    # 输出两个参数
    alpha, beta = results.params
    # 输出t值
    alpha_t, beta_t = results.tvalues
    # 输出p值
    alpha_p, beta_p = results.pvalues
    return alpha, alpha_t, alpha_p


# ff3估计α
def fama_french3(ret, rf, risk_premium, SMB, HML):
    # 合并列
    X = np.column_stack((risk_premium, SMB))
    X = np.column_stack((X, HML))
    # 添加常数列
    X = sm.add_constant(X)
    # 进行拟合
    model = sm.OLS(ret - rf, X)
    # 拟合结果
    results = model.fit()
    # 输出四个参数
    alpha, beta, gama, tau = results.params
    # 输出t值
    alpha_t, beta_t, gama_t, tau_t = results.tvalues
    # 输出p值
    alpha_p, beta_p, gama_p, tau_p = results.pvalues
    return alpha, alpha_t, alpha_p


# 输出多空收益的各个指标值
def test_duokong():
    # 读取数据
    ret = pd.read_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\duokong.xlsx')
    RF = pd.read_csv('D:\\study\\Quantitative_investment\\单因子选股\\RF.csv')
    ff3 = pd.read_csv('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\ff30.csv')
    # 通过读取的数据，给我们需要的参数赋值（时间段1997.1-2018.9，因为ff3文件中数据只给出了这一时间段的）
    rf = np.array(RF.iloc[12:, 1])
    risk_premium = np.array(ff3.iloc[12:, 1])
    SMB = np.array(ff3.iloc[12:, 2])
    HML = np.array(ff3.iloc[12:, 3])
    duokong = np.array(ret.iloc[0:252, 1])
    # 计算平均值
    duokong_mean = np.mean(duokong)
    print("average:{}".format(duokong_mean))
    # nw调整t检验
    duokong_nwttest, duokong_nwp = nwttest_1samp(duokong, 0, L=1)
    print("nwttest:{}, nwp:{}".format(duokong_nwttest, duokong_nwp))
    # 正常t检验
    duokong_ttest, duokong_p = stats.ttest_1samp(duokong, 0)
    print("ttest:{}, p:{}".format(duokong_ttest, duokong_p))
    # 计算标准差（为计算夏普比率做准备）
    duokong_std = np.std(duokong)
    # 计算夏普比率
    duokong_sharp = (duokong_mean - np.mean(rf)) / duokong_std
    print("sharp:{}".format(duokong_sharp))
    # 计算CAPM调整α
    CAPM_alpha, CAPM_alpha_t, CAPM_alpha_p = CAPM(duokong, rf, risk_premium)
    print("CAPM_alpha:{}, CAPM_alpha_t:{}, CAPM_alpha_p:{}".format(CAPM_alpha, CAPM_alpha_t, CAPM_alpha_p))
    # 计算ff3调整α
    ff3_alpha, ff3_alpha_t, ff3_alpha_p = fama_french3(duokong, rf, risk_premium, SMB, HML)
    print("ff3_alpha:{}, ff3_alpha_t:{}, ff3_alpha_p:{}".format(ff3_alpha, ff3_alpha_t, ff3_alpha_p))


# 计算十分位投资的各个指标
def test_touzi():
    # 读取数据
    ret = pd.read_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\ret.xlsx')
    RF = pd.read_csv('D:\\study\\Quantitative_investment\\单因子选股\\RF.csv')
    ff3 = pd.read_csv('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\ff30.csv')
    # 通过读取的数据，给我们需要的参数赋值（时间段1997.1-2018.9，因为ff3文件中数据只给出了这一时间段的）
    rf = np.array(RF.iloc[12:, 1])
    risk_premium = np.array(ff3.iloc[12:, 1])
    SMB = np.array(ff3.iloc[12:, 2])
    HML = np.array(ff3.iloc[12:, 3])
    # 定义数组存储数据
    ret_x_mean = [0] * 10
    ret_x_nwttest = [0] * 10
    ret_x_ttest = [0] * 10
    ret_x_sharp = [0] * 10
    CAPM_alpha = [0] * 10
    CAPM_alpha_t = [0] * 10
    CAPM_alpha_p = [0] * 10
    ff3_alpha = [0] * 10
    ff3_alpha_t = [0] * 10
    ff3_alpha_p = [0] * 10
    # 计算各个指标（跟上一个函数命名类似）
    for i in range(10):
        ret_x = np.array(ret.iloc[0:252, i + 1])
        ret_x_mean[i] = np.mean(ret_x)
        ret_x_nwttest[i], ret_x_nwp = nwttest_1samp(ret_x, 0, L=1)
        ret_x_ttest[i], ret_x_p = stats.ttest_1samp(ret_x, 0)
        ret_x_std = np.std(ret_x)
        ret_x_sharp[i] = (ret_x_mean[i] - np.mean(rf)) / ret_x_std
        CAPM_alpha[i], CAPM_alpha_t[i], CAPM_alpha_p[i] = CAPM(ret_x, rf, risk_premium)
        ff3_alpha[i], ff3_alpha_t[i], ff3_alpha_p[i] = fama_french3(ret_x, rf, risk_premium, SMB, HML)
    # 输出结果
    print("十个分位数投资组合的平均收益序列：{}".format(ret_x_mean))
    print("十个分位数投资组合的nw调整t检验序列：{}".format(ret_x_nwttest))
    print("十个分位数投资组合的t检验序列：{}".format(ret_x_ttest))
    print("十个分位数投资组合的夏普比率序列：{}".format(ret_x_sharp))
    print("十个分位数投资组合的CAPM调整α序列：{}".format(CAPM_alpha))
    print("十个分位数投资组合的CAPM调整α的t检验序列：{}".format(CAPM_alpha_t))
    print("十个分位数投资组合的CAPM调整α的t检验的p值序列：{}".format(CAPM_alpha_p))
    print("十个分位数投资组合的ff3调整α序列：{}".format(ff3_alpha))
    print("十个分位数投资组合的ff3调整α的t检验序列：{}".format(ff3_alpha_t))
    print("十个分位数投资组合的ff3调整α的t检验的p值序列：{}".format(ff3_alpha_p))
    all = []
    all.append(ret_x_mean)
    all.append(ret_x_nwttest)
    all.append(ret_x_ttest)
    all.append(ret_x_sharp)
    all.append(CAPM_alpha)
    all.append(CAPM_alpha_t)
    all.append(CAPM_alpha_p)
    all.append(ff3_alpha)
    all.append(ff3_alpha_t)
    all.append(ff3_alpha_p)
    all = pd.DataFrame(all)
    all.to_excel('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\all.xlsx')


# 读取文件
df = pd.read_csv('D:\\study\\Quantitative_investment\\基于机器学习的多因子量化投资系统\\predictions_1.csv')

# 问题十三
# AM_201812 = stock_codes(df, 251)
# top_10 = AM_201812.iloc[:, -10:].columns.tolist()
# last_10 = AM_201812.iloc[:, 0: 10].columns.tolist()
# print('TOP10:')
# print(top_10)
# print('LAST10:')
# print(last_10)
# find_r(top_10)
# print('**********************************************************************************************************')
# find_r(last_10)

# 问题十四
# AM = np.array(stock_codes(df, 251).columns)
# AM = AM.tolist()
# print(AM)
# touzi, duokong = calc(AM, 263)
# print("投资组合的收益为：{}".format(touzi))
# print("多空组合的收益为：{}".format(duokong))

# 问题七
# 这一问运行时间长是正常的，因为需要进行的计算量特别大，大约需要将近一个小时
# time_series(df)
# 绘图
# plot_time_series()

# 问题十五
test_duokong()
test_touzi()
