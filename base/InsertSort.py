#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
插入排序

从第一个元素开始，该元素可以认为已经被排序；
取出下一个元素，在已经排序的元素序列中从后向前扫描；
如果该元素（已排序）大于新元素，将该元素移到下一位置；
重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
将新元素插入到该位置后；
重复步骤2~5。
"""


"""
方法一：内层使用while循环
"""
def insert_sort(lst: list):
    for i in range(1, len(lst)):
        # pre_index = i - 1
        while lst[i] < lst[i - 1] and i - 1 >= 0:
            lst[i], lst[i - 1] = lst[i - 1], lst[i]
            i -= 1

    return lst


"""
方法二：内层使用for循环
"""
def insert_sort2(lst: list):
    for i in range(1, len(lst)):
        # [i-1,0)
        for j in range(i, 0, -1):
            if lst[j-1] > lst[j]:
                lst[j-1], lst[j] = lst[j], lst[j-1]
            else:
                break
    return lst


if __name__ == '__main__':
    arr = [14, 2, 34, 43, 21, 19]
    # sorted_lst = insert_sort(arr)
    sorted_lst = insert_sort2(arr)
    print(sorted_lst)

