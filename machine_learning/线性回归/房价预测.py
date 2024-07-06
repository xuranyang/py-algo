from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S")

"""
通过线性回归实现 波士顿房价预测
"""

dataset = datasets.load_boston()  # 加载数据集,共506条记录
x: np.ndarray = dataset.data  # 获取特征数据,共13个维度
y: np.ndarray = dataset.target  # 获取标签数据

# 通过dir函数查看对象内的所有的属性和方法名称
# dir(dataset) # ['DESCR', 'data', 'feature_names', 'filename', 'target']
# logging.info(f"Feature Shape:{x.shape}")  # (506, 13)
# logging.info(f"Label Shape:{y.shape}")  # (506,)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 特征工程-数据标准化(可选)
scaler = StandardScaler()
# 标准化后的数据,具有均值为0和标准差为1的特性
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 创建线性回归模型
model = LinearRegression()

# 训练模型的参数
model.fit(x_train, y_train)

# 预测
y_test_predict = model.predict(x_test)

# 计算MSE均方误差
mse = mean_squared_error(y_test, y_test_predict)

print(f"模型的系数为:{model.coef_}")
print(f"模型的偏置为:{model.intercept_}")
print(f"测试集的均方误差:{mse}")

plt.style.use('ggplot')  # 设置画图风格
plt.rcParams.update({'font.size': 16})  # 设置字体大小
plt.figure(figsize=(10, 6))
plt.plot(y_test_predict, linewidth=3, label="predict")
plt.plot(y_test, linewidth=3, label="real")
plt.legend(loc='best')
plt.xlabel('test data points')
plt.ylabel('target value')
plt.show()
