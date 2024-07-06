from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S")

"""
通过线性回归实现 波士顿房价预测
特征维度对应的中文解释：
    CRIM：城镇人均犯罪率
    ZN： 占地面积超过2.5万平方英尺的住宅用地比例
    INDUS：城镇上非零售业务地区的 比例
    CHAS：虚拟变量；如果土地在查尔斯河，取值1；否则为0
    NOX：一氧化氮浓度
    RM：平均每个居民房数
    AGE：在1940年之前建成的所有者占用单位的比例
    DIS： 与波士顿的5个就业中心之间的加权距离
    RAD： 辐距离住房最近的公路入口编号
    TAX：每10,000美元的全额物业税
    PTRATIO：城镇师生比例大小
    B：1000(Bk-0.63)^2,其中 Bk 指代城镇中黑人的比例
    LSTAT：全部人口中地位较低人群的百分数大小
    MEDV：目标变量，以1000美元来进行计算的自由住房的中位数大小
"""


def load_data():
    """
    加载数据
    :return: (feature_data, label_data)
    """
    dataset = datasets.load_boston()  # 加载数据集,共506条记录
    feature_data: np.ndarray = dataset.data  # 获取特征数据,共13个维度
    label_data: np.ndarray = dataset.target  # 获取标签数据

    # 特征维度名称 ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
    logging.info(f"Feature Names:{dataset.feature_names}")
    logging.info(f"Feature Shape:{feature_data.shape}")  # (506, 13)
    logging.info(f"Label Shape:{label_data.shape}")  # (506,)

    return feature_data, label_data


def split_data(x, y, test_size=0.3):
    """
    划分训练集和测试集
    :param x: 特征数据
    :param y: 标签数据
    :param test_size: 测试集的占比,默认0.3
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def data_process(x_train, x_test, y_train, y_test):
    """
    自定义数据处理
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    # 特征工程-数据标准化(可选)
    scaler = StandardScaler()
    # 标准化后的数据,具有均值为0和标准差为1的特性
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def train(x_train, y_train):
    """
    模型训练
    使用训练集进行训练
    :param x_train:
    :param y_train:
    :return:
    """
    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型的参数
    model.fit(x_train, y_train)

    return model


def predict(model, x_test):
    """
    预测: 使用测试集进行预测
    :param model: 进行预测的模型
    :param x_test:
    :param y_test:
    :return:
    """
    # 预测
    y_test_predict = model.predict(x_test)
    return y_test_predict


def evaluate(model, x_test, y_test, y_test_predict):
    """
    模型效果评估
    :return:
    """
    # 计算MSE均方误差
    mse = mean_squared_error(y_test, y_test_predict)
    # 回归任务,返回的是R^2分数,越接近1 则拟合程度越好
    score = model.score(x_test, y_test)
    print(f"模型的系数为:{model.coef_}")
    print(f"模型的偏置为:{model.intercept_}")
    print(f"测试集的均方误差:{mse}")
    print(f"测试集得分:{score}")


def plt_show(label_test, label_predict):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')  # 设置画图风格
    plt.rcParams.update({'font.size': 16})  # 设置字体大小
    plt.figure(figsize=(10, 6))
    plt.plot(label_predict, linewidth=3, label="predict")
    plt.plot(label_test, linewidth=3, label="real")
    plt.legend(loc='best')
    plt.xlabel('test data points')
    plt.ylabel('target value')
    plt.show()


def save_model(model, filename):
    from sklearn.externals import joblib
    """
    保存模型
    :return:
    """
    joblib.dump(model, filename=filename)


def load_model(filename):
    """
    加载模型
    :return:
    """
    from sklearn.externals import joblib
    model = joblib.load(filename)
    return model


def main():
    feature, label = load_data()  # 加载数据
    split_datas = split_data(feature, label)  # 数据划分
    x_train, x_test, y_train, y_test = data_process(*split_datas)  # 数据处理

    # model = load_model('./linear_regression_boston.pkl')  # 加载已有模型
    model = train(x_train, y_train)  # 模型训练
    y_test_predict = predict(model, x_test)  # 测试集预测
    evaluate(model, x_test, y_test, y_test_predict)  # 预测结果的效果评估
    plt_show(y_test, y_test_predict)  # 可视化展示

    # save_model(model, './linear_regression_boston.pkl') # 保存模型


if __name__ == '__main__':
    main()
