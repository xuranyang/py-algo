import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import os

"""
训练数据集：train-images-idx3-ubyte.gz
训练结果集：train-labels-idx1-ubyte.gz

测试数据集：t10k-images-idx3-ubyte.gz
测试结果集：t10k-labels-idx1-ubyte.gz
"""


class LeNet(nn.Module):
    """
    LeNet神经网络输入图像大小必须为32x32，且所用卷积核大小固定为5x5

    INPUT（输入层）：输入图像尺寸为32x32，且是单通道灰色图像。
    C1（卷积层）：使用6个5x5大小的卷积核，步长为1，卷积后得到6张28×28的特征图。
    S2（池化层）：使用了6个2×2 的平均池化，池化后得到6张14×14的特征图。
    C3（卷积层）：使用了16个大小为5×5的卷积核，步长为1，得到 16 张10×10的特征图。
    S4（池化层）：使用16个2×2的平均池化，池化后得到16张5×5 的特征图。
    F5（全连接层）：使用120个大小为5×5的卷积核，步长为1，卷积后得到120张1×1的特征图。
    F6（全连接层）：输入维度120，输出维度是84（对应7x12 的比特图）。
    OUTPUT（输出层）：使用高斯核函数，输入维度84，输出维度是10（对应数字 0 到 9）。
    """

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(  # MNIST数据集图像大小为28x28，而LeNet输入为32x32，故需填充
            # C1+S2层共六个卷积核，故out_channels=6
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # S2层使用最大池化
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),  # C3+S4,16个卷积核
            nn.MaxPool2d(kernel_size=2, stride=2),  # S4层使用最大池化
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),  # F5, 120个卷积核
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),  # F6, 120->84
            nn.Linear(in_features=84, out_features=10)  # OUTPUT, 84->10
        )

    def forward(self, img):
        # 正向传播函数
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def data_explore():
    mnist_train = torchvision.datasets.MNIST(root='../data/', train=True, download=True,
                                             transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(root='../data/', train=False, download=True,
                                            transform=transforms.ToTensor())
    print('mnist_train基本信息为：', mnist_train)
    print('-----------------------------------------')
    print('mnist_test基本信息为：', mnist_test)
    print('-----------------------------------------')
    img, label = mnist_train[0]
    print('mnist_train[0]图像大小及标签为：', img.shape, label)


def load_data_fashion_mnist(batch_size, resize=None, root='../data'):
    import sys
    """Use torchvision.datasets module to download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    # transform = torchvision.transforms.Compose(trans)
    # mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    # mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # set the model to evaluation mode (disable dropout)
                net.eval()
                # get the acc of this batch
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # change back to train mode
                net.train()

            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, optimizer, num_epochs, device=try_gpu()):
    net = net.to(device)  # model,适配GPU/CPU
    print("training on", device)
    loss_fn = torch.nn.CrossEntropyLoss()  # 设置损失函数为交叉熵损失函数
    batch_count = 0
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            imgs = X.to(device)  # 适配GPU/CPU
            labels = y.to(device)  # 适配GPU/CPU

            y_hat = net(imgs)  # y_hat为outputs
            loss = loss_fn(y_hat, labels)  # 计算损失函数
            optimizer.zero_grad()  # 清空之前梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == labels).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)

        print(
            f'epoch {epoch + 1} : loss {train_loss_sum / batch_count:.3f}, train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}')
        save_model(model=net, epoch=epoch + 1, model_name='LeNet-5.pth')


def save_model(model, epoch, model_dir='saved_models', model_name='LeNet-5.pth'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, model_path)
    print(f"Model saved at: {model_path}")


def load_model(model, model_dir='saved_models', model_name='LeNet-5.pth'):
    model_path = os.path.join(model_dir, model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {model_path}")
    return model, checkpoint['epoch']


def main():
    batch_size = 256
    lr, num_epochs = 0.9, 10

    # 初始化模型对象
    le_net = LeNet()
    optimizer = torch.optim.SGD(le_net.parameters(), lr=lr)

    # load data
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    # train
    train(le_net, train_iter, test_iter, optimizer, num_epochs)


def evaluate():
    model = LeNet().to(device=try_gpu())
    model, epoch = load_model(model)
    _, test_iter = load_data_fashion_mnist(batch_size=256)

    correct = 0
    total = 0
    for inputs, labels in tqdm(test_iter):
        inputs, labels = inputs.to(device=try_gpu()), labels.to(device=try_gpu())
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")


def show_data():
    model = LeNet().to(device=try_gpu())
    model, epoch = load_model(model)
    model.eval()  # 设置模型为评估模式

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    _, test_iter = load_data_fashion_mnist(batch_size=5)
    for idx in range(5):
        img, real_label = test_iter.dataset[idx]
        outputs = model(img.to(device=try_gpu()).unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        predict_label = predicted.item()
        print(img.shape)  # [1, 28, 28]
        print(f"实际:{real_label},预测:{predict_label}")

        axes[idx].imshow(img.cpu().squeeze(), cmap='gray')
        axes[idx].set_title(f"预测:{predict_label}")
        axes[idx].axis('off')

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    # evaluate()
    show_data()
