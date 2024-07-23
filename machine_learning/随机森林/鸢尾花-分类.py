# 导入所需库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    # 加载鸢尾花数据集
    iris_dataset = load_iris()
    return iris_dataset


def process_data(dataset):
    x = dataset.data
    y = dataset.target
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def train(x_train, y_train):
    # 使用随机森林进行分类
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    # 预测测试集
    y_predict = model.predict(x_test)
    return y_predict


def evaluate_model(y_test, y_predict):
    # 输出准确率
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy准确率: {accuracy:.2f}")

    # 输出分类报告和混淆矩阵
    print("Classification Report:")
    print(classification_report(y_test, y_predict, target_names=load_iris().target_names))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_predict)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=load_iris().target_names,
                yticklabels=load_iris().target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def show(model):
    # 绘制特征重要性
    feature_importances = model.feature_importances_
    features = load_iris().feature_names
    sns.barplot(x=feature_importances, y=features)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest")
    plt.show()


def main():
    dataset = load_data()
    x_train, x_test, y_train, y_test = process_data(dataset)
    model = train(x_train, y_train)
    y_predict = predict(model, x_test)
    evaluate_model(y_test, y_predict)
    show(model)


if __name__ == '__main__':
    main()
