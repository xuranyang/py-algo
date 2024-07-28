from math import ceil

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

"""
digits 手写数字数据集
['data', 'target', 'target_names', 'images', 'DESCR']

共1797个样本，每个样本包括8*8像素的图像和一个[0, 9]整数的标签
target_names 为[0~9]

data.shape为(1797, 64)
    data中的每个样本为 1*64
images.shape为(1797, 8, 8)
    images中的每个样本为 8*8

target.shape为(1797, 1)
"""


def create_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    models = {
        'LR': LogisticRegression(),  # 逻辑回归
        'DT': DecisionTreeClassifier(),  # 决策树
        'RF': RandomForestClassifier(100),  # 随机森林
        'KNN': KNeighborsClassifier(n_neighbors=3),  # K近邻
        'SVM': SVC(gamma='scale')  # 支持向量机
    }

    return models


def load_data():
    digits_dataset = datasets.load_digits()
    data = digits_dataset.data
    target = digits_dataset.target
    return data, target


def main():
    # load_data 加载数据集
    data, target = load_data()
    # split_data 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

    # init model 初始化模型
    model = create_models().get('SVM')
    # train_model 训练模型
    model = model.fit(X_train, y_train)
    # evaluate 评估模型
    score = model.score(X_test, y_test)  # predict + accuracy_score
    print(f"ACC准确率:{score}")

    # 抽样预测展示
    show_predict(model, X_test, y_test, top_n=10)


def predict_result(model, X_test, y_test, idx):
    """
    预测单个样本的结果
    :param model:
    :param X_test: 测试集-特征
    :param y_test: 测试集-标签
    :param idx: 测试集的第几个样本
    :return:
    """


def show_predict(model, X_test, y_test, top_n=10):
    cols = 5
    rows = ceil(top_n / cols)
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        # sharex=True,
        sharey=True,
    )
    axs = ax.flatten()
    for idx, ax in enumerate(axs[:top_n]):
        y_predict = model.predict(X_test[idx].reshape(1, 64))
        y_real = y_test[idx]
        print(f"预测数字:{y_predict}|实际数字:{y_real}")
        ax.matshow(X_test[idx].reshape(8, 8))
        ax.set_title(f"{y_real}|{y_predict}", pad=20)
    plt.tight_layout()
    plt.show()


def show_example_data(rows=4, cols=5):
    """
    仅数据展示
    :param rows:
    :param cols:
    :return:
    """
    digits_dataset = datasets.load_digits()
    top_n = rows * cols
    # 默认展示前二十个数据的图像
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        # sharex=True,
        sharey=True,
    )
    axs = ax.flatten()  # 将axs从 2*5 转换成 1*10
    for idx, ax in enumerate(axs[:top_n]):
        # ax.imshow(digits_dataset.data[idx].reshape((8, 8)), cmap='Greys', interpolation='nearest')
        # ax.imshow(digits_dataset.images[idx], cmap='Greys', interpolation='nearest')
        ax.matshow(digits_dataset.images[idx])
        ax.set_title(f"Num:{digits_dataset.target[idx]}", pad=20)  # pad:标题与图表顶部的距离，单位为点（points）

    # 调整子图间的间距以避免标题重叠
    # plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # show_example_data(4, 5)
    main()
