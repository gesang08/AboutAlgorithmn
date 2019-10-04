#!/usr/bin/env python3
# coding=utf-8

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import SR1
from scipy.optimize import BFGS
def createFuncdata(mod=1):
    if mod not in [1,2,3,4]:
        return
    x = np.linspace(-5.0, 5.0,100)
    if mod == 1:
        y = 2 * x + 3 + np.random.randn(len(x))  # 加入噪音  # 产生一次函数数据
    elif mod == 2:
        y = 4 * np.power(x, 2) - 4 * x + 5 + np.random.randn(len(x))  # 加入噪音
    elif mod==3:
        y = np.power(x, 3) + 9 + np.random.randn(len(x))  # 加入噪音
    else:
        x = np.linspace(-5, 5)
        y = np.power(x,2) + 1 + np.random.randn(len(x))  # 加入噪音
    return x,y

bounds = Bounds([0,-0.5],[1.0,2.0])
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

def cons_f(x):
    return [x[0]**2 + x[1], x[0]**2 - x[1]]
def cons_J(x):
    return [[2*x[0], 1], [2*x[0], -1]]
def cons_H(x, v):
    return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

x0 = np.array([0.5, 0])
def rosen(x):
    return 100 * (x[1]-x[0]**2)**2 +(1-x[0])**2
res = minimize(rosen, x0, method='trust-constr', jac="2-point", hess=SR1(),
               constraints=[linear_constraint, nonlinear_constraint],
                               options={'verbose': 1}, bounds=bounds)
print(res.x)