import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from NWttest import nwttest_1samp
from scipy import stats
import statsmodels.api as sm


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
    ret = pd.read_excel('D:\\study\\Quantitative_investment\\test\\duokong_1.xlsx')
    RF = pd.read_excel('D:\\study\\Quantitative_investment\\test\\rf.xlsx')
    ff3 = pd.read_csv('D:\\study\\Quantitative_investment\\test\\ff3.csv')
    # 通过读取的数据，给我们需要的参数赋值（时间段1997.1-2018.9，因为ff3文件中数据只给出了这一时间段的）
    rf = np.array(RF.iloc[0:261, 1])
    risk_premium = np.array(ff3.iloc[0:261, 1])
    SMB = np.array(ff3.iloc[0:261, 2])
    HML = np.array(ff3.iloc[0:261, 3])
    duokong = np.array(ret.iloc[0:261, 1])
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
    duokong_sharp = (duokong_mean - np.mean(rf))/ duokong_std
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
    ret = pd.read_excel('D:\\study\\Quantitative_investment\\test\\ret_1.xlsx')
    RF = pd.read_excel('D:\\study\\Quantitative_investment\\test\\rf.xlsx')
    ff3 = pd.read_csv('D:\\study\\Quantitative_investment\\test\\ff3.csv')
    # 通过读取的数据，给我们需要的参数赋值（时间段1997.1-2018.9，因为ff3文件中数据只给出了这一时间段的）
    rf = np.array(RF.iloc[0:261, 1])
    risk_premium = np.array(ff3.iloc[0:261, 1])
    SMB = np.array(ff3.iloc[0:261, 2])
    HML = np.array(ff3.iloc[0:261, 3])
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
        ret_x = np.array(ret.iloc[0:261, i + 1])
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


test_duokong()
test_touzi()
