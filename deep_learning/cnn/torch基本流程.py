import torchvision.datasets
import torch.utils.data
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
import torch.nn.functional as F

"""
PyTorch训练一个网络的基本流程5步法
step1. 加载数据
step2. 定义网络
step3. 定义损失函数和优化器
step4. 训练网络，循环4.1到4.6直到达到预定epoch数量
    – step4.1 加载数据
    – step4.2 初始化梯度
    – step4.3 计算前馈
    – step4.4 计算损失
    – step4.5 计算梯度
    – step4.6 更新权值
step5. 保存权重
"""


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        input -> conv1 -> relu -> max_pool -> conv2 -> relu -> max_pool -> fc1 -> relu -> fc2 -> relu -> fc3 -> output
        :param x:
        :return:
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    """
    训练
    :return:
    """

    """1.加载数据"""
    # ToTensor转化为pytorch的基本数据类型Tensor
    # Normalize归一化
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # 第一次运行download=True
    # 训练集
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

    # 测试集
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    # classes = (
    #     'plane', 'car', 'bird', 'cat', 'deer',
    #     'dog', 'frog', 'horse', 'ship', 'truck'
    # )

    """2.定义网络"""
    my_net = MyNet()

    """3.定义损失函数and优化器"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_net.parameters(), lr=1e-3, momentum=0.9)

    """cuda加速"""
    device = ['gpu' if torch.cuda.is_available() else 'cpu']
    if device == 'gpu':
        criterion.cuda()
        my_net.to(device)
        # Net.cuda()      #多GPU 请用 DataParallel方法

    """4.训练网络"""
    print('开始训练')
    for epoch in range(3):
        runing_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, label = data  # 1.数据加载
            if device == 'gpu':
                inputs = inputs.cuda()
                label = label.cuda()
            optimizer.zero_grad()  # 2.初始化梯度
            output = my_net(inputs)  # 3.计算前馈
            loss = criterion(output, label)  # 4.计算损失
            loss.backward()  # 5.计算梯度
            optimizer.step()  # 6.更新权值

            runing_loss += loss.item()
            if i % 20 == 19:
                print('epoch:', epoch, 'loss', runing_loss / 20)
                runing_loss = 0.0

    print('训练完成')
    """5.保存模型参数"""
    torch.save(my_net.state_dict(), 'cifar10_net.pth')

    """6.模型测试评估"""
    evaluate_test(my_net, test_loader, device)


def evaluate_test(net, test_loader, device):
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            net.eval()
            out = net(test_x)
            _, y_hat = torch.max(out.data, dim=1)
            total += test_x.size(0)
            correct += (y_hat == test_y).sum().item()
    print('accuracy:', correct / total)


if __name__ == '__main__':
    train()
