import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

"""
MNIST数据集: 包含了0-9共10类手写数字图片,每张图片都做了尺寸归一化,都是28x28大小的灰度图
6万个训练数据 + 1万个测试数据

Fashion-MNIST数据集：包含不同商品的正面图片，涵盖 10 个类别(T恤,鞋子等类别)的图像,每张图像的尺寸为28x28像素‌
70000 张灰度图像(6万个训练数据 + 1万个测试数据)
"""


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1个输入通道(灰度图像=1，RGB图像=3) 和 16个输出通道（卷积核数量）
        # 卷积核大小为3 或 3*3 + 步长为1(卷积核滑动的步幅) + 填充为1(保持输入输出尺寸一致)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # 第一次池化后，从 28x28 变成 14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层卷积，输入是16个通道，输出是32个通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # 第一次池化后，从 14x14 变成 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 经过两次池化后，图像尺寸会缩小; 原始是28x28，第一次池化后变成14x14，第二次变成7x7; 通道数是32;
        # 所以全连接层的输入特征数是 32*7*7，输出特征数设置为128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        # 再经过一个全连接层，输入特征数是128，输出特征数设置为10(分别对应数字0~9共10个类别)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# 数据加载函数
def load_data():
    # 数据标准化处理（均值0.1307，标准差0.3081）
    # 根据官方数据，MNIST的均值是0.1307，标准差是0.3081。所以归一化用这两个值
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transform)
    return train_dataset, test_dataset


def get_data_loader():
    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


# 训练函数
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播(求导)
        optimizer.step()  # 更新参数权重

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# 测试函数
def test(model, device, test_loader, criterion):
    # 将模型设置为评估模式
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main(is_save=True):
    # 主程序
    # 初始化配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loader()
    # 初始化模型、损失、优化器
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    if is_save:
        # 保存模型参数
        torch.save(model.state_dict(), 'mnist_cnn_3.pt')


# 可视化预测单个MNIST样本
def predict_single(model, device, test_dataset, index=-1):
    """
    预测单个MNIST样本
    :param model: 训练好的模型
    :param device: 使用的设备
    :param test_dataset: 测试数据集
    :param index: 样本索引（默认随机选择）
    :return: (预测值, 真实值, 图像数据)
    """
    # 随机选择一个样本
    if index == -1:
        index = np.random.randint(0, len(test_dataset))

    # 获取数据和标签
    image, true_label = test_dataset[index]

    # 添加batch维度并送到设备
    image = image.unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted_label = torch.argmax(output).item()

    # 将图像数据转为numpy格式用于显示
    image_np = image.squeeze().cpu().numpy()

    return predicted_label, true_label, image_np


def predict_batch(batch_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)  # 定义相同的模型结构
    model.load_state_dict(torch.load('mnist_cnn_3.pt'))  # 加载模型参数
    _, test_dataset = load_data()

    # 随机预测5个样本
    for _ in range(batch_size):
        predicted, true_label, image = predict_single(model, device, test_dataset)
        print(f"预测: {predicted} | 真实: {true_label}")


# 可视化图片
def show_image(predicted, true_label, image):
    # 可视化结果
    # plt.imshow(image, cmap='gray')
    plt.imshow(image)
    plt.title(f"Predicted: {predicted}, True: {true_label}")
    plt.axis('off')
    plt.show()

    print(f"\n预测结果: {predicted}")
    print(f"真实标签: {true_label}")


def visualize_predict():
    # 示例预测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)  # 定义相同的模型结构
    model.load_state_dict(torch.load('mnist_cnn_3.pt'))  # 加载模型参数
    _, test_dataset = load_data()
    predicted, true_label, image = predict_single(model, device, test_dataset, index=-1)
    # 可视化结果
    show_image(predicted, true_label, image)


if __name__ == '__main__':
    main(is_save=True)
    predict_batch(batch_size=5)
    visualize_predict()
