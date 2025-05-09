import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检查保存模型和数据的文件夹是否存在，如果不存在则创建
save_dir = 'saved_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 使用CUDA进行计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义保存模型函数
def save_model(model, epoch, model_dir='saved_models', model_name='simple_cnn_model.pth'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, model_path)
    print(f"Model saved at: {model_path}")


# 定义加载模型函数
def load_model(model, model_dir='saved_models', model_name='simple_cnn_model.pth'):
    model_path = os.path.join(model_dir, model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {model_path}")
    return model, checkpoint['epoch']


# 定义可视化函数
def visualize_predictions(model, test_loader, classes, num_images=5):
    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    with torch.no_grad():
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for idx in range(num_images):
        axes[idx].imshow(images[idx].cpu().squeeze(), cmap='gray')
        axes[idx].set_title(f"True: {classes[labels[idx].item()]}\nPredicted: {classes[predicted[idx].item()]}",
                            fontsize=12)
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


def visualize_dataset(dataset_loader, classes, num_images=5):
    data_iter = iter(dataset_loader)
    images, labels = data_iter.next()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for idx in range(num_images):
        axes[idx].imshow(images[idx].cpu().squeeze(), cmap='gray')
        axes[idx].set_title(f"Label: {classes[labels[idx].item()]}", fontsize=12)
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


# 在训练过程中保存模型和损失值
def train(model, train_loader, criterion, optimizer, epochs=5, save_interval=1):
    losses = []  # 记录每个epoch的损失值
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 适配GPU/CPU
            optimizer.zero_grad()  # 清空之前梯度
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()

        # 计算并记录每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        # 打印每个epoch的平均损失
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        # 每隔一定的epoch保存模型
        if (epoch + 1) % save_interval == 0:
            save_model(model, epoch + 1)

    # 保存损失数据
    np.savetxt(os.path.join(save_dir, 'losses.txt'), np.array(losses))


# 加载保存的模型进行测试
def predict(model, test_loader, classes):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

    # 可视化模型在测试集中的预测结果
    visualize_predictions(model, test_loader, classes)


def main():
    # 训练模型
    criterion = nn.CrossEntropyLoss()  # 设置损失函数为交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    visualize_dataset(train_loader, classes)
    train(model, train_loader, criterion, optimizer, epochs=10, save_interval=5)
    show_loss()


def show_loss():
    # 绘制损失曲线图
    losses = np.loadtxt(os.path.join(save_dir, 'losses.txt'))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()


# 定义标签类别
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 数据集处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 创建模型并将其移到CUDA上
model = SimpleCNN().to(device)

if __name__ == '__main__':
    # main()
    # 加载保存的模型进行测试
    model, epoch = load_model(model)
    predict(model, test_loader, classes)
