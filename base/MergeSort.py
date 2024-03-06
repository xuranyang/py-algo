#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
归并排序

图解可以参考 https://www.cnblogs.com/chengxiao/p/6194356.html
分割：递归地把当前序列平均分割成两半
集成：在保持元素顺序的同时将上一步得到的子序列集成到一起（归并）。

归并排序的最优、最坏时间复杂度均为O(nlogn)。稳定
平均时间复杂度也为O(nlogn)
"""


def merge_sort(arr):
    arr_len = len(arr)
    if arr_len <= 1:
        return arr

    # 将列表分成两半
    mid = arr_len // 2
    # 递归地对左右两部分进行归并排序
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    print(left)
    print(right)
    print("======")
    # 合并排序后的左右两部分
    return merge(left, right)


def merge(left_arr, right_arr):
    result = []
    left_idx = right_idx = 0

    while left_idx < len(left_arr) and right_idx < len(right_arr):
        # 如果当前左边数组的元素 小于等于 当前右边数组的元素 则 放入最终结果列表 并将左边数组的下标+1
        if left_arr[left_idx] <= right_arr[right_idx]:
            result.append(left_arr[left_idx])
            left_idx += 1
        # 如果当前右边数组的元素 小于 当前左边数组的元素 则 放入最终结果列表 并将右边数组的下标+1
        # 简单写可以直接else
        # else
        elif right_arr[right_idx] < left_arr[left_idx]:
            result.append(right_arr[right_idx])
            right_idx += 1
    # 说明左边数组的数字已经全部取完,将右边数组剩余的数字全部放进最终结果即可
    if left_idx == len(left_arr):
        for i in right_arr[right_idx:]:
            result.append(i)
    # 否则说明右边数组的数字已经全部取完,将左边数组剩余的数字全部放进最终结果即可
    # 简单写可以直接else
    # else
    elif right_idx == len(right_arr):
        for j in left_arr[left_idx:]:
            result.append(j)

    return result


if __name__ == '__main__':
    arr = [14, 2, 34, 43, 21, 19]
    # arr = [3, 1, 7, 2, 5, 4, 9, 6, 8]
    sorted_arr = merge_sort(arr)
    print(sorted_arr)
