#!/usr/bin/env python3
# coding=utf-8

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize
from scipy import stats
from scipy.optimize import LinearConstraint
import time
from test import createFuncdata
# 读取电脑自带的字体文件，simsun为黑体-->解决plt画图上中文乱码问题
simsun = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=12)
FLAG=False
TIME = time.strftime('%Y{y}%m{m}%d{d}%H{h}%M{f}%S{s}').format(y='', m='', d='', h='', f='', s='')
############################可修改的API接口###########################
file_path = './data/test.xls'  # 文件的修改
result_save_path = './data/result%s.xls'% TIME
POWER = 0  # 想获取的拟合函数:1、2、3分别对应一、二、三次函数;0表示自动获取偏差(防止欠拟合)和方差(防止过拟合)最小的拟合函数
limit_theta0 = -900  # 0:方程常数项>0限制
limit_theta2 = 1  # 1:二次方程，二次项系数>1
###################################################################
class GetData:
    def read_excel(self, file_path):
        """
        读取excel文件
        :param file_path: 文件路径
        :return: x,y,colNames列表
        """
        x, y, bais =[], [], []
        with xlrd.open_workbook(filename=file_path) as data:
            table = data.sheets()[0]  # 通过索引顺序获取表
            colNames=table.row_values(0)
            for i in range(table.nrows-1):
                x.append(table.row_values(i+1, start_colx=0, end_colx=1)[0])
                y.append(table.row_values(i+1, start_colx=1, end_colx=2)[0])
                bais.append(table.row_values(i + 1, start_colx=2, end_colx=3)[0])
        return x, y, colNames,bais

class GradientDescent:
    def costReg(self, theta, X, y, lamda):
        """
        # 带正则化的代价函数
        :param theta: 拟合的参数
        :param X: input
        :param y: label
        :param lamda: 正则化参数
        :return: 代价值
        """
        cost = np.sum(np.power((X@theta-y.flatten()),2)) #theta是一维数据，y.flatten()进行围堵匹配
        reg = theta[1:]@theta[1:] * lamda
        return (cost + reg) / (2 * len(X))

    def gradientReg(self, theta, X, y, lamda):  # 正则化的梯度
        """
        正则化的梯度
        :param theta: 拟合参数
        :param X: input
        :param y: label
        :param lamda: 正则化参数
        :return: 梯度
        """
        grad = (X @ theta - y.flatten()) @ X
        reg = lamda * theta
        reg[0] = 0  # theta0不参与正则化
        return (grad + reg) / len(X)

    def train_modal(self, X, y, lamda,bais):
        """
        梯度下降方法进行优化拟合
        :param X: input
        :param y: label
        :param lamda: 正则化参数
        :return: 拟合出的theta
        """
        theta = np.ones(X.shape[1])
        y = y.flatten()

        # cons = ({'type':'ineq',
        #          'fun':lambda X: -(np.power(X @ theta, 2) - np.multiply(bais, np.power(y,2))),
        #          'jac':lambda X: 2 * (X.dot((X.dot(theta) - y)))
        #          })

        linear_constraint = LinearConstraint(X, -np.inf, y + bais*y)  # 偏差bais的10%限制
        res = minimize(fun=self.costReg,
                       x0=theta,
                       args=(X, y, lamda),
                       method='trust-constr',
                       jac=self.gradientReg,constraints=linear_constraint,options={'disp':False})
        # res = minimize(fun=costReg,options={'maxiter': 250, 'verbose': 3}
        #                x0=theta,
        #                args=(X, y, lamda),
        #                method='BFGS',
        #                jac=gradientReg)
        # print(res.success)
        # print(res.fun)
        # print(res.x)
        # print(res.message)
        return res.x

    def train_modal1(self, X, y, lamda,bais):
        """
        梯度下降方法进行优化拟合
        :param X: input
        :param y: label
        :param lamda: 正则化参数
        :return: 拟合出的theta
        """
        theta = np.ones(X.shape[1])
        res = minimize(fun=self.costReg,
                       x0=theta,
                       args=(X, y, lamda),
                       method='BFGS',
                       jac=self.gradientReg)
        return res.x

    def normalEquation(self, attrMat, label):
        """
        正规方程法inv(X.T@X)@X.T@y求theta
        :param attrMat:特征矩阵-->X
        :param label:标签-->y
        :return:theta
        """
        return np.linalg.pinv(attrMat.T @ attrMat) @ attrMat.T @ label  # 用伪逆矩阵防止不可逆情况

def ploy_feature(X, power):
    """
    获取多项式特征
    :param X:
    :param power:
    :return:
    """
    for i in range(2, power+1):  # 从2次幂开始取
        X=np.insert(X,obj=X.shape[1],values=np.power(X[:,1],i),axis=1)
    return X

def get_means_stds(X):
    """
    获取均值和标准差
    :param X:
    :return:
    """
    means = np.mean(X, axis=0)
    stds = np.std(X,axis=0)
    return means,stds

def feature_norm(X,means,stds):
    means = np.tile(means, (X.shape[0],1))
    stds= np.tile(stds, (X.shape[0],1))
    X[:,1:] = (X[:,1:] - means[:,1:])/stds[:,1:]
    return X

def get_best_lamda(X,y,bais):
    lamdas = np.arange(0.0,1.0,0.01) + np.arange(1.0,101.0,1)  # 可以自动选取最好的lamda进行最小偏差和方差拟合
    train_cost = []
    # validation_cost = []  # 理论最好用验证集找到最好的lamda
    for lamda in lamdas:
        theta = GradientDescent().train_modal1(X,y,lamda,bais)
        tc = GradientDescent().costReg(theta,X,y,lamda)
        train_cost.append(tc)
    # plt.plot(lamdas,train_cost)
    # plt.show()

    # best_lamda = lamdas[np.argmin(validation_cost)]
    best_lamda = lamdas[np.argmin(train_cost)]
    return best_lamda

def significance_test(X,y,theta): # 显著性检验
    sigma = np.sqrt(np.sum(y - X @ theta) ** 2) / X.shape[0]
    t = theta[1:] * np.sqrt(np.sum((X[:, 1:] - np.mean(X[:, 1:], axis=0)) ** 2, axis=0)) / sigma
    p = stats.t.sf(t, X.shape[0])
    if min(p) < 0.05:
        print('the linear regression between x and y is significant')
    else:
        print('the linear regression between x and y is not significant')

def pltResult(x, y, theta, expression, titleName, _x=None,x_dot=None):
    fig = plt.figure(num="拟合结果图")
    plt.scatter(x, y, label="Training data")
    plt.title(titleName, fontproperties=simsun)
    plt.xlabel("x", fontproperties=simsun)
    plt.ylabel("y", fontproperties=simsun)
    # fig.autofmt_xdate()

    if theta.shape[0] == 2:
        plt.plot(x, x * theta[1] + theta[0], 'r--',label="Prediction")
    elif theta.shape[0] == 3 or 4:
        plt.plot(_x, x_dot @ theta, 'r--',label="Prediction")
    else:
        print('No support the function of fitting')
        return
    plt.text(min(x),max(y)/2,expression)
    plt.legend(loc=2)
    plt.show()

def calcBais(X,y,theta):
    y = y.flatten()
    predict = X @ theta
    bais = np.fabs(y - predict)/y
    return bais

def saveExcel(data):
    dp = pd.DataFrame(data)
    try:
        dp.to_excel(result_save_path,index=False)
        print('结果保存至%s文件，请查阅' % result_save_path)
    except:
        print('请关闭%s，进行结果保存' % result_save_path)

def oneTimesFit(X,y,bais):
    # theta = np.ones(X.shape[1])
    # costValue = costReg(theta, X, y, lamda)
    # print(costValue)
    # gradientValue = gradientReg(theta, X, y, lamda)
    # print(gradientValue)
    lamda = get_best_lamda(X,y,bais)
    one_times_theta = GradientDescent().train_modal(X,y,lamda=lamda,bais=bais)

    # print(X @ one_times_theta)
    pred_bais = calcBais(X, y, one_times_theta)
    goalNums = np.sum(pred_bais > bais)

    # lossMean = np.mean([GradientDescent().costReg(one_times_theta, X[:i, :], y[:i], lamda=lamda) for i in range(1, X.shape[0] + 1)])
    if POWER == 0 and not FLAG:
        return goalNums
    expression = 'y = %s * x + %s' % (one_times_theta[1], one_times_theta[0])

    if one_times_theta[0] > limit_theta0:
        print(expression)
        pltResult(X[:, 1:], y, one_times_theta,expression,titleName='一次函数拟合')
        significance_test(X,y,one_times_theta)
        return X @ one_times_theta, pred_bais
    else:
        print('the constant term named theta0=%s less than %s in 1-order function fitting.' % (one_times_theta[0],limit_theta0))
        print('please choose other POWER.')
        return [0]*X.shape[0],[0]*X.shape[0]

def moreTimesFit(X,y,bais,power):  # 可以实现多次函数拟合
    X_ploy = ploy_feature(X, power)
    means, stds = get_means_stds(X_ploy)
    X_norm = feature_norm(X_ploy, means, stds)
    lamda = get_best_lamda(X_norm,y,bais)
    ploy_theta = GradientDescent().train_modal(X_norm, y, lamda,bais)
    # print(X_norm @ ploy_theta)
    # ploy_theta = normalEquation(X_norm,y)
    pred_bais = calcBais(X_ploy, y, ploy_theta)
    goalNums = np.sum(pred_bais > bais)
    # lossMean = np.mean([GradientDescent().costReg(ploy_theta,X_norm,y,lamda) for i in range(1, X.shape[0]+1)])
    if POWER == 0 and not FLAG:
        return goalNums

    if ploy_theta[0] > limit_theta0:
        _x = np.arange(min(X[:, 1]), max(X[:, 1]) + 1, 1)
        x_dot = _x.reshape(_x.shape[0], 1)
        x_dot = np.insert(x_dot, obj=0, values=np.ones(x_dot.shape[0]), axis=1)
        x_dot = ploy_feature(x_dot, power)
        x_dot = feature_norm(x_dot, means, stds)
        # _x = X
        # x_dot = X_norm

        if power == 2:  # 二次函数拟合
            if ploy_theta[2] > limit_theta2:
                expression = 'y = %s * x^2 + \n%s * x + %s' % (ploy_theta[2], ploy_theta[1], ploy_theta[0])
                print(expression)
                pltResult(X[:, 1], y, ploy_theta, expression, titleName='二次函数拟合',_x=_x,x_dot=x_dot)

                significance_test(X_norm,y,ploy_theta)

            else:
                print('the constant term named theta2=%s less than %s in secondary-order function fitting.' % (ploy_theta[2],limit_theta2))
                print('please choose other POWER.')
                return
        elif power == 3:
            expression = 'y = %s * x^3 + %s * x^2 +\n %s * x + %s' % (ploy_theta[3],ploy_theta[2], ploy_theta[1], ploy_theta[0])
            print(expression)
            pltResult(X[:, 1], y, ploy_theta, expression, titleName='三次函数拟合',_x=_x,x_dot=x_dot)
            significance_test(X_norm,y,ploy_theta)
        else:
            print('No support the function of fitting')
            return
    else:
        print('the constant term named theta0 less than 0 in %s-order function fitting.' % power)
        print('please choose other POWER.')
        return [0]*X.shape[0],[]*X.shape[0]
    return X_norm @ ploy_theta, pred_bais

def fitMain():

    X, y, colNames, bais = GetData().read_excel(file_path)
    # X,y = createFuncdata(3)
    # bais = 0.1 * np.ones(y.shape[0])
    X = np.reshape(np.array(X), (np.array(X).shape[0], 1))
    y = np.reshape(np.array(y), (np.array(y).shape[0], 1))
    bais = np.array(bais)
    X = np.insert(arr=X,obj=0,values=np.ones(X.shape[0]),axis=1)
    m = X.shape[0]
    predict1, pred_bais1, predict2, pred_bais2, predict3, pred_bais3 = [0]*m, [0]*m, [0]*m, [0]*m, [0]*m, [0]*m

    if POWER == 1:
        predict1,pred_bais1=oneTimesFit(X,y,bais)
    elif POWER == 2:
        predict2, pred_bais2 =moreTimesFit(X,y,bais,power=2)
    elif POWER == 3:
        predict3, pred_bais3 =moreTimesFit(X, y, bais,power=3)
    elif POWER == 0:
        a = [oneTimesFit(X,y,bais), moreTimesFit(X,y,bais,power=2),moreTimesFit(X, y, bais,power=3)]
        if min(a) != 0:
            print('No suitable fit function for the limit of bais.')
        minIndex = np.argmin(a)
        global FLAG
        FLAG = True
        if minIndex == 0 and FLAG:
            predict1, pred_bais1 = oneTimesFit(X, y,bais)
        elif minIndex == 1:
            predict2, pred_bais2=moreTimesFit(X, y,bais, power=2)
        elif minIndex == 2:
            predict3, pred_bais3=moreTimesFit(X, y,bais, power=3)
    else:
        print('No support the function of fitting by the %s POWER' % POWER)
    saveExcel(data={'X':X[:,1],'y':y.flatten(),'bais':bais,
                    'predict1': predict1, 'pred_bais1': pred_bais1,
                    'predict2': predict2, 'pred_bais2': pred_bais2,
                    'predict3': predict3, 'pred_bais3': pred_bais3})

if __name__ == '__main__':
    fitMain()