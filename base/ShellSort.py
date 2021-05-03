#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
希尔排序（缩小增量排序）
插入排序的改进版
"""


def shell_sort(lst):
    # 初始化增量gap=length/2 向下取整
    gap = len(lst) // 2
    while gap >= 1:
        # 遍历每一个组
        for i in range(gap, len(lst)):
            # 同一组内,进行插入排序
            while (i - gap) >= 0:
                if lst[i] < lst[i - gap]:
                    lst[i], lst[i - gap] = lst[i - gap], lst[i]
                    i -= gap
                else:
                    break
        # 缩小增量gap为原来的一半
        gap = gap // 2

    return lst


if __name__ == '__main__':
    arr = [14, 2, 34, 43, 21, 19]
    sorted_arr = shell_sort(arr)
    print(sorted_arr)
