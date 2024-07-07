import pandas as pd

"""
共12个特征：
    PassengerId：乘客编号
    Survived：是否生存，1表生存，0表示遇难
    Pclass：舱位等级，分为一等舱、二等舱、三等舱
    Name：乘客姓名
    Sex：性别，Male或Female
    Age：年龄
    SibSp：兄弟姐妹、堂兄弟姐妹人数
    Parch：父母与子女个数
    Ticket：船票信息（上面记载着座位号）
    Fare：票价
    Cabin：客舱
    Embarked：登船港口

train.csv中的这12列数据，有9列数据是完整的，即有891条记录
    Embarked这一列，数据缺失了两条；
    Age这一列，差了一百多条数据；
    Cabin这一列，数据很不完整，只有204条记录。
"""


def load_data() -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv('data/titanic/train.csv')
    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # 剔除无关的feature
    data = data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
    # 缺失值处理
    # 用平均值填充
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    # 用众数填充
    data['Embarked'] = data['Embarked'].fillna('S')

    # one hot编码
    data_dummy = pd.get_dummies(data[['Sex', 'Embarked']])
    data = pd.concat([data, data_dummy], axis=1)
    # 删除编码前的特征 性别与登船港
    data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
    return data


def split_data(data, test_size=0.3, random_state=0):
    from sklearn.model_selection import train_test_split

    # x为特征,y为标签
    x = data.drop('Survived', axis=1)
    y = data['Survived']

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def train(x_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    y_predict = model.predict(x_test)
    return y_predict


def evaluate(model, x_test, y_test, y_predict):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    print("测试集acc准确率:", accuracy_score(y_test, y_predict))
    # 两种方法一样,都是计算acc
    # print("测试集acc准确率:", model.score(x_test, y_test))
    print(classification_report(y_test, y_predict))


def main():
    # 加载数据
    data = load_data()
    # 特征工程
    clean_data = process_data(data)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = split_data(clean_data)
    # 训练模型
    model = train(x_train, y_train)
    # 预测结果
    y_predict = predict(model, x_test)
    # 模型效果评估
    evaluate(model, x_test, y_test, y_predict)
    # 保存 or 加载模型
    # save_model(model, filename)
    # model = load_model(filename)


if __name__ == '__main__':
    main()
