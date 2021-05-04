"""
计数排序
基础版
开一个长度为 maxValue+1 的数组，然后分配。
扫描一遍原始数组，以 当前值 作为下标，将该下标的计数器增1。
收集。扫描一遍计数器数组，按顺序把值收集起来。

优化版
开一个长度为 maxValue-minValue+1 的数组，然后分配。
扫描一遍原始数组，以 当前值-minValue 作为下标，将该下标的计数器增1。
收集。扫描一遍计数器数组，按顺序把值收集起来。
"""


def count_sort(arr: list):
    """
    基础版
    :param arr:
    :return:
    """
    # 选择一个最大的数
    max_num = max(arr)
    # 创建一个元素全是0的列表, 当做桶
    bucket = [0] * (max_num + 1)
    # 把所有元素放入桶中, 即把对应元素个数加一
    for i in arr:
        bucket[i] += 1
    # 存储排序好的元素
    sort_nums = []
    # 取出桶中的元素
    for j in range(len(bucket)):
        if bucket[j] != 0:
            for y in range(bucket[j]):
                sort_nums.append(j)
    return sort_nums


def count_sort_optimize(arr: list):
    """
    优化版
    :param arr:
    :return:
    """
    # 选择一个最大的数
    max_num = max(arr)
    min_num = min(arr)
    # 创建一个元素全是0的列表, 当做桶
    bucket = [0] * (max_num - min_num + 1)
    # 把所有元素放入桶中, 即把对应元素个数加一
    for i in arr:
        bucket[i - min_num] += 1
    # 存储排序好的元素
    sort_nums = []
    # 取出桶中的元素
    for j in range(len(bucket)):
        if bucket[j] != 0:
            for y in range(bucket[j]):
                sort_nums.append(j)
    return sort_nums


if __name__ == '__main__':
    li = [14, 2, 34, 43, 21, 19]
    # sorted_li = count_sort(li)
    sorted_li = count_sort_optimize(li)
    print(sorted_li)
