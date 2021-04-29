#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random

"""
选择排序
第一轮：找到最小的数放置到第一位
第二轮：从第二个数开始找，找到第二小的数放到第二位
......以此类推
"""


def select_sort(lst: list):
    for i in range(len(lst)):
        min_index = i
        for j in range(i + 1, len(lst)):
            if lst[min_index] > lst[j]:
                min_index = j
        lst[i], lst[min_index] = lst[min_index], lst[i]

    return lst


if __name__ == '__main__':
    arr = list(range(10))
    random.shuffle(arr)
    arr = [14, 2, 34, 43, 21, 19]
    # print(arr)
    print("---------")
    sorted_arr = select_sort(arr)
    print(sorted_arr)
