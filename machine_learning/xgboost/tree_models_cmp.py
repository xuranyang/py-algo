from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型
models = {
    "RandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False),
    "LightGBM": LGBMClassifier(n_estimators=10, random_state=42)
}

# 训练和评估每个模型
accuracy_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 绘制各模型的准确率对比
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.title("Model Comparison on Iris Dataset")
plt.show()
