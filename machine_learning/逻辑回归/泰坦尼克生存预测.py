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

