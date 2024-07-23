from matplotlib import pyplot as plt

"""
现在有10根蜡烛, 每根蜡烛的质量为1, 经过10分钟后全部燃烧完毕(质量为0)；

假设经过t分钟后：
    当 0<=t<=10 时,当前的质量 g(t)=1-0.1t
    当 t<0时,当前的质量g(t)为1
    当 t>10时,当前的质量g(t)为0

假设现在：
第t分钟 点燃f(t)根蜡烛
第0分钟 点燃1根蜡烛
第4分钟 点燃4根蜡烛
第5分钟 点燃3根蜡烛
第9分钟 点燃2根蜡烛

用f(t)表示在第t分钟点燃的蜡烛数量，即：
f(0)=1
f(4)=4
f(5)=3
f(9)=2



假设现在为x=7,在第7分钟时:
 f(t)根 第t分钟点燃,燃烧了7-t分钟,每根还剩 g(x-t)=g(7-t)
    1根 第0分钟点燃,燃烧了7分钟,每根蜡烛剩 g(7-0)=g(7)=1-0.1*7=0.3
    4根 第4分钟点燃,燃烧了3分钟,每根蜡烛剩 g(7-4)=g(3)=1-0.1*3=0.7
    3根 第5分钟点燃,燃烧了2分钟,每根蜡烛剩 g(7-5)=g(2)=1-0.1*2=0.8
    2根 第9分钟点燃,燃烧了-2分钟,每根蜡烛剩 g(7-9)=g(-2)=1
    
所以当x=7时,
h(x)=(f*g)(x)
    =f(0)*g(7-0) + f(4)*g(7-4) + f(5)*g(7-5) +f(9)*g(7-9)
    =f(0)*g(7) + f(4)*g(3) + f(5)*g(2) +f(9)*g(-2)
    =1*0.3 + 4*0.7 + 3*0.8 + 2*1
    =0.3 + 2.8 + 2.4 + 2
    =7.5
"""


def g(t):
    if 0 <= t <= 10:
        return 1 - 0.1 * t
    elif t < 0:
        return 1
    elif t > 10:
        return 0


def f(t):
    """
    在第t分钟时,点燃的蜡烛数量
    :param t:
    :return:
    """
    if t == 0:
        return 1
    elif t == 4:
        return 4
    elif t == 5:
        return 3
    elif t == 9:
        return 2
    else:
        return 0


def show_convolution():
    """
    卷积
    :return:
    """
    label_x_fgx = []
    label_y_fgx = []

    # x = 7
    for x in range(0, 20):
        # 表示卷积
        fg_x = 0
        for t in range(11):
            f_t = f(t)
            g_xt = g(x - t)
            fg_x += f_t * g_xt
        print(x, fg_x)
        label_x_fgx.append(x)
        label_y_fgx.append(fg_x)

    # plt.figure(figsize=(len(label_x), len(label_y)))
    plt.plot(label_x_fgx, label_y_fgx, color='red', linestyle='-', marker='o')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('第x分钟')
    plt.ylabel('蜡烛的剩余质量')
    plt.legend(['卷积:f(t)*g(x-t)'])
    plt.show()


def get_all_ax():
    label_x = []
    label_ft = []
    label_gt = []
    label_gtx = []
    label_gxt = []
    x = 7  # 令 x=7
    for t in range(-3, 11):
        f_t = f(t)
        g_t = g(t)
        g_tx = g(t - x)
        g_xt = g(x - t)
        label_x.append(t)
        label_ft.append(f_t)
        label_gt.append(g_t)
        label_gtx.append(g_tx)
        label_gxt.append(g_xt)
    return {
        'f(t)': (label_x, label_ft),
        'g(t)': (label_x, label_gt),
        'g(t-7)': (label_x, label_gtx),
        'g(7-t)': (label_x, label_gxt)
    }


def show_all():
    # 创建1个图形和4个轴
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    label_x = []
    label_ft = []
    label_gt = []
    label_gtx = []
    label_gxt = []
    x = 7  # 令 x=7
    # 创建一些数据进行绘制
    for t in range(-3, 11):
        f_t = f(t)
        g_t = g(t)
        g_tx = g(t - x)
        g_xt = g(x - t)
        label_x.append(t)
        label_ft.append(f_t)
        label_gt.append(g_t)
        label_gtx.append(g_tx)
        label_gxt.append(g_xt)

    # 在第一个轴上绘制第一张图
    ax1.plot(label_x, label_ft)
    ax1.set_title('f(t)')  # 设置第1张图的标题

    # 在第二个轴上绘制第二张图
    ax2.plot(label_x, label_gt)
    ax2.set_title('g(t)')  # 设置第2张图的标题

    ax3.plot(label_x, label_gtx)
    ax3.set_title('g(t-7)')  # 设置第3张图的标题

    ax4.plot(label_x, label_gxt)
    ax4.set_title('g(7-t)')  # 设置第4张图的标题

    plt.tight_layout()  # 当有多个子图时，可以使用该语句保证各子图标题不会重叠
    # 显示图形
    plt.show()


def show_part(ax1_name, ax2_name):
    # 创建1个图形和2个轴
    fig, (ax1, ax2) = plt.subplots(2)

    # 在第一个轴上绘制第一张图
    x, y = get_all_ax().get(ax1_name)
    ax1.plot(x, y)
    ax1.set_title(ax1_name)  # 设置第1张图的标题

    # 在第二个轴上绘制第二张图
    x, y = get_all_ax().get(ax2_name)
    ax2.plot(x, y)
    ax2.set_title(ax2_name)  # 设置第2张图的标题

    plt.tight_layout()  # 当有多个子图时，可以使用该语句保证各子图标题不会重叠
    # 显示图形
    plt.show()


if __name__ == '__main__':
    show_convolution()
    # show_all()
    # show_part('f(t)', 'g(t)')
    # show_part('f(t)', 'g(t-7)')
    show_part('f(t)', 'g(7-t)')
