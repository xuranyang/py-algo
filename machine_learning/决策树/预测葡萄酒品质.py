from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import graphviz

"""
    alcohol : 酒精
    malic_acid : 苹果酸
    ash : 灰
    alcalinity_of_ash : 灰的碱性
    magnesium : 镁
    total_phenols : 总酚
    flavanoids : 类黄酮
    nonflavanoid_phenols : 非黄烷类酚类
    proanthocyanins : 花青素
    color_intensity : 颜色强度
    hue : 色调
    od280/od315_of_diluted_wines : od280/od315稀释葡萄酒
    proline : 脯氨酸
"""


def load_data():
    wine_data = load_wine()
    # print(wine_data.data) # 数据集
    # print(wine_data.data.shape)  # (178, 13) 共178条记录
    # print(wine_data.feature_names)  # 13种特征
    # print(wine_data.target)  # 结果集:共3种类型:0 1 2

    data: np.ndarray = wine_data.data
    label: np.ndarray = wine_data.target
    feature_names: list = wine_data.feature_names
    return data, label, feature_names


def split_data(data, label, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=test_size)
    return x_train, x_test, y_train, y_test


def clf_train(x_train, y_train, max_depth):
    # criterion -> entropy 使用信息熵;gini 使用基尼系数
    # 分类树
    clf_model = tree.DecisionTreeClassifier(criterion='entropy'
                                            , random_state=30
                                            , max_depth=max_depth  # 指定最大深度,避免过拟合
                                            )

    # reg_model = tree.DecisionTreeRegressor()  # 回归树,如:波士顿房价预测
    clf_model.fit(x_train, y_train)

    return clf_model


def evaluate(model, x_test, y_test):
    y_predict = model.predict(x_test)
    # acc_score = accuracy_score(y_test, y_predict)
    score = model.score(x_test, y_test)  # 分类的score是acc,回归的score是R2
    print("score:", score)
    return score


def main():
    feature, label, feature_names = load_data()
    x_train, x_test, y_train, y_test = split_data(feature, label, test_size=0.3)
    clf_model = clf_train(x_train, y_train, max_depth=None)
    evaluate(clf_model, x_test, y_test)

    dot_data = tree.export_graphviz(clf_model
                                    , feature_names=feature_names
                                    , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                    , filled=True, rounded=True
                                    )
    graph = graphviz.Source(dot_data.replace('helvetica', '"Microsoft YaHei"'), encoding='utf-8')
    graph.render(view=True)


def post_pruning():
    """
    循环 max_depth -> 1~10 不同的决策树深度,找到最合适的深度-后剪枝
    :return:
    """
    scores = []
    for i in range(1, 11):
        feature, label, feature_names = load_data()
        x_train, x_test, y_train, y_test = split_data(feature, label, test_size=0.3)
        clf_model = clf_train(x_train, y_train, max_depth=i)
        score = evaluate(clf_model, x_test, y_test)
        scores.append(score)

    print("最合适的max_depth:", scores.index(max(scores)) + 1)
    plt.plot(range(1, 11), scores, color='green', label='max_depth')
    plt.legend()
    plt.show()


def cross_validate():
    """
    通过交叉验证
    同时观察 模型在 训练集和测试集 上的表现，避免过拟合
    :return:
    """
    # 通过学习曲线，观察在不同深度下模型拟合的状况
    train_score = []
    test_score = []
    for i in range(10):
        feature, label, feature_names = load_data()
        x_train, x_test, y_train, y_test = split_data(feature, label, test_size=0.2)
        clf = tree.DecisionTreeClassifier(criterion='entropy'
                                          , random_state=25
                                          , max_depth=i + 1  # 指定最大深度,避免过拟合
                                          )
        clf.fit(x_train, y_train)
        score_train = clf.score(x_test, y_test)
        # 使用全量的数据集求取交叉验证的均值
        score_test = cross_val_score(clf, feature, label, cv=10).mean()
        train_score.append(score_train)
        test_score.append(score_test)
    # 打印最大的测试分数,以及所在的索引
    print(test_score)
    # print(max(test_score), test_score.index(max(test_score)) + 1)
    print("当最大深度为", test_score.index(max(test_score)) + 1, "时,测试集的分数最高:", max(test_score))
    # 画图
    plt.plot(range(1, 11), train_score, color='orange', label='train')
    plt.plot(range(1, 11), test_score, color='green', label='test')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    # post_pruning()
    cross_validate()
