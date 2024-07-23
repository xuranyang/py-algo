import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor


def load_data():
    """
    数据集的一行代表一个街区的普查
        MedInc - 每个街区的收入中位数。
        HouseAge - 每个街区的房屋年龄中位数。
        AveRooms - 每户人家的平均房间数量。
        AveBedrms - 每户人家的平均卧室数量。
        Population - 街区人口。
        AveOccup - 平均每户人家的居住成员数量。
        Latitude - 街区纬度。
        Longitude - 街区经度。
        MedHouseVal - 房屋中位价值（以千元为单位）。
    :return:
    """
    from sklearn.utils import Bunch

    dataset: Bunch = fetch_california_housing()
    # ['data', 'target', 'feature_names', 'DESCR']
    # print(dataset.keys())
    # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    print(dataset.feature_names)
    # print(dataset.data.shape)  # (20640, 8)
    feature = dataset.data
    label = dataset.target
    print(label)

    return feature, label


def process_data(x, y, test_size=0.3):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def train(x_train, y_train):
    # model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    y_predict = model.predict(x_test)
    return y_predict


def evaluate(y_test, y_predict):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_test, y_predict)  # 均方误差, Sum[(y_test-y_predict)^2]
    mae = mean_absolute_error(y_test, y_predict)  # 平均绝对误差 SUM[ abs(y_test-y_predict) ]/ y.size
    r2 = r2_score(y_test, y_predict)  # R2分数的取值范围在0到1之间，越接近1表示模型拟合效果越好，越接近0表示模型拟合效果越差
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")


def show(model, y_test, y_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 绘制预测值与实际值的散点图
    # plt.scatter(y_test, y_pred, alpha=0.5)
    # plt.xlabel("Actual Values")
    # plt.ylabel("Predicted Values")
    # plt.title("Actual vs Predicted Values (Random Forest Regression)")
    # plt.show()

    # 绘制特征重要性
    feature_importances = model.feature_importances_
    features = fetch_california_housing().feature_names
    sns.barplot(x=feature_importances, y=features)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest Regression")
    plt.show()


def main():
    feature, label = load_data()
    x_train, x_test, y_train, y_test = process_data(feature, label)
    model = train(x_train, y_train)
    y_predict = predict(model, x_test)
    evaluate(y_test, y_predict)
    show(model, y_test, y_predict)


if __name__ == '__main__':
    main()
