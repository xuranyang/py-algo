#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random

"""
二分查找
"""


def binary_search(opt_val, lst: list, left_index, right_index):
    if left_index <= right_index:
        mid = int((left_index + right_index) / 2)

        if lst[mid] == opt_val:
            return mid
        # opt_val比lst[mid]小 说明在mid坐标的左边
        elif opt_val < lst[mid]:
            right_index = mid - 1
            return binary_search(opt_val, lst, left_index, right_index)
        # opt_val比lst[mid]大 说明在mid坐标的右边
        elif opt_val > lst[mid]:
            left_index = mid + 1
            return binary_search(opt_val, lst, left_index, right_index)
    else:
        return 'None'


if __name__ == '__main__':
    the_lst = [x for x in range(10)]
    random.shuffle(the_lst)
    print(the_lst)
    the_lst = sorted(the_lst)
    print(the_lst)

    pos = binary_search(8, the_lst, 0, len(the_lst) - 1)
    print("POS[%s]:%s" % (pos, the_lst[pos]))
