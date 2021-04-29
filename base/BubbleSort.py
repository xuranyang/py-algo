#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random

"""
冒泡排序

依次比较相邻的两个数，将比较小的数放在前面，比较大的数放在后面。
在第一趟比较完成后，比较第二趟的时候，最后一个数是不参加比较的。
在第二趟比较完成后，在第三趟的比较中，最后两个数是不参与比较的。
......以此类推
"""


def bubble_sort(lst: list):
    for i in range(0, len(lst) - 1):
        for j in range(0, len(lst) - (i + 1)):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


def bubble_sort_optimize(lst: list):
    """
    优化版 如果发现一轮下来没有交换顺序，与上一轮一致，说明已经完成排序
    可以直接返回 不再继续遍历
    :param lst:
    :return: sorted_list
    """
    for i in range(0, len(lst) - 1):
        tag = False
        for j in range(0, len(lst) - (i + 1)):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                tag = True
        if tag is False:
            print("times:%d" % i)
            return lst
    return lst


if __name__ == '__main__':
    arr = list(range(10))
    random.shuffle(arr)
    # arr = [14, 2, 34, 43, 21, 19]
    print(arr)
    print("---------")
    sorted_arr = bubble_sort(arr)
    print(sorted_arr)
    # sorted_arr = bubble_sort_optimize(arr)
    # print(sorted_arr)
