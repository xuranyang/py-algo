"""
conda install -c anaconda py-xgboost

参数	作用
num_boost_round	集成算法中弱分类器数量，对Boosting算法而言为实际迭代次数
eta	            Boosting算法中的学习率，影响弱分类器结果的加权求和过程
objective	    选择需要优化的损失函数
base_score	    初始化预测结果H0的设置
max_delta_step	一次迭代中所允许的最大迭代值
gamma	        乘在叶子数量前的系数，放大可控制过拟合
lambda	        L2正则化系数，放大可控制过拟合
alpha	        L1正则化系数，放大可控制过拟合
"""
# lightgbm原生接口
import xgboost as xgb
# 基于scikit-learn接口
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston, load_breast_cancer, load_wine
import warnings

warnings.simplefilter("ignore")

bonston = load_boston()
cancer = load_breast_cancer()
wine = load_wine()


def cancer_xgboost_predict():
    data_train, data_test, target_train, target_test = train_test_split(cancer.data, cancer.target, test_size=0.2,
                                                                        random_state=0)

    params = {
        'eta': 0.02,  # lr
        'max_depth': 6,
        'min_child_weight': 3,  # 最小叶子节点样本权重和
        'gamma': 0,  # 指定节点分裂所需的最小损失函数下降值。
        'subsample': 0.7,  # 控制对于每棵树，随机采样的比例
        'colsample_bytree': 0.3,  # 用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
        'lambda': 2,  # L2正则化系数，放大可控制过拟合
        'objective': 'binary:logistic',  # 选择需要优化的损失函数
        'eval_metric': 'auc',
        'silent': True,
        'nthread': -1
    }

    xgb_train = xgb.DMatrix(data_train, target_train)
    xgb_test = xgb.DMatrix(data_test, target_test)
    xgb_model = xgb.train(dtrain=xgb_train, params=params)
    xgb_predict = xgb_model.predict(xgb_train)
    xgb_predict[xgb_predict > .5] = 1
    xgb_predict[xgb_predict <= .5] = 0

    xgboost_show(params, xgb_train)


def wine_xgboost_predict():
    data_train, data_test, target_train, target_test = train_test_split(wine.data, wine.target, test_size=0.2,
                                                                        random_state=0)

    params = {
        'eta': 0.02,  # lr
        'num_class': 3,
        'max_depth': 6,
        'min_child_weight': 3,  # 最小叶子节点样本权重和
        'gamma': 0,  # 指定节点分裂所需的最小损失函数下降值。
        'subsample': 0.7,  # 控制对于每棵树，随机采样的比例
        'colsample_bytree': 0.3,  # 用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
        'lambda': 2,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'silent': True,
        'nthread': -1
    }

    xgb_train = xgb.DMatrix(data_train, target_train)
    xgb_test = xgb.DMatrix(data_test, target_test)
    xgb_model = xgb.train(dtrain=xgb_train, params=params)
    xgb_predict = xgb_model.predict(xgb_train)
    xgb_test_pred = xgb_model.predict(xgb_test)

    xgboost_show(params, xgb_train)


def boston_xgboost_predict():
    data_train, data_test, target_train, target_test = train_test_split(bonston.data, bonston.target, test_size=0.2,
                                                                        random_state=0)

    params = {
        'eta': 0.02,  # lr
        'max_depth': 6,
        'min_child_weight': 3,  # 最小叶子节点样本权重和
        'gamma': 0,  # 指定节点分裂所需的最小损失函数下降值。
        'subsample': 0.7,  # 控制对于每棵树，随机采样的比例
        'colsample_bytree': 0.3,  # 用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
        'lambda': 2,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        #   'silent': True,
        'nthread': -1}

    xgb_train = xgb.DMatrix(data_train, target_train)
    xgb_test = xgb.DMatrix(data_test, target_test)
    xgb_model = xgb.train(dtrain=xgb_train, params=params, num_boost_round=100)
    xgb_train_predict = xgb_model.predict(xgb_train)
    train_mae_score = mean_absolute_error(xgb_train_predict, target_train)
    print('train mae score:', train_mae_score)
    xgb_test_predict = xgb_model.predict(xgb_test)
    test_mae_score = mean_absolute_error(xgb_test_predict, target_test)
    print('test mae score:', test_mae_score)

    xgboost_show(params, xgb_train)


def xgboost_show(params, xgb_train):
    result = xgb.cv(params, xgb_train, num_boost_round=300, nfold=5, seed=2022)
    # print(result)
    plt.figure(dpi=90)
    plt.plot(result["train-{}-mean".format(params["eval_metric"])])
    plt.plot(result["test-{}-mean".format(params["eval_metric"])])
    plt.legend(["train", "test"])
    plt.title("xgboost 5 fold cv")
    plt.show()


if __name__ == '__main__':
    # cancer_xgboost_predict()
    # wine_xgboost_predict()
    boston_xgboost_predict()
