#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
快速排序
分为4个步骤
找一个基准数（参照数）
从右往左找比基准数小的数与左坐标交换
从左往右找比基准数大的数与右坐标交换
左、右坐标相遇时，基准数与相遇坐标交换
"""

"""
arr = [6, 8, 7, 9, 5]
mid = arr[0] = 6

[6, 8, 7, 9, 5]
 l           r
[5, 8, 7, 9, 5]
 l           r
[5, 8, 7, 9, 8]
    l        r
[5, 8, 7, 9, 8]
    lr
[5, mid, 7, 9, 8]
    lr
[5, 6, 7, 9, 8]
    lr

分为[5,6] 和 [7,9,8]

arr1=[5,6]
mid = arr1[0]=5
[5, 6]
 l  r
[5, 6]
 lr
[mid,6]
[5,6]


arr2=[7,9,8]
mid = arr2[0]=7

[7, 9, 8]
 l     r
[7, 9, 8]
 lr
[mid, 9, 8]
[7, 9 , 8]

arr3 = [9,8]
mid = arr3[0] = 9

[9, 8]
 l  r
[8, 8]
 l  r
[8, 9]
    lr
[8, mid]
    lr
[8, 9]


[5,6] [7] [8,9]
"""


"""
快排写法1
"""
def quick_sort(li, start, end):
    """
    :param li: 要排序的list
    :param start: 排序的起始位置
    :param end: 排序的结束位置
    :return: list
    """
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = li[left]
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        while left < right and li[right] >= mid:
            right -= 1
        li[left] = li[right]
        # 让左边游标往右移动，目的是找到大于等于mid的值，放到right游标位置
        while left < right and li[left] < mid:
            left += 1
        li[right] = li[left]
    # while结束后，把mid放到中间位置，left=right
    li[left] = mid
    # 递归处理左边的数据
    quick_sort(li, start, left - 1)
    # 递归处理右边的数据
    quick_sort(li, left + 1, end)


"""
快排写法2
这种写法更好理解
1.右游标 从右往左找到第一个小于arr[0]的数字,与左游标交换位置
2.左游标 从左往右找到第一个大于等于arr[0]的数字,与右游标交换位置
以此类推 直到左右游标到同一位置为止,此时该位置左边的值 都小于游标,右边的值都大于等于游标
对左右两边继续递归 直到最后结束
"""
def quick_sort2(li, start, end):
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = li[left]
    #     while left < right:
    # 让右边游标往左移动，目的是找到小于mid的值，交换左右游标的值
    while left < right and li[right] >= mid:
        right -= 1
    li[left], li[right] = li[right], li[left]
    # 让左边游标往右移动，目的是找到大于等于mid的值，交换左右游标的值
    while left < right and li[left] < mid:
        left += 1
    li[left], li[right] = li[right], li[left]
    # mid 取第0个元素时 可以不需要再设置
    # li[left] = mid
    li[left] = mid
    # 递归处理左边的数据
    quick_sort2(li, start, left - 1)
    # 递归处理右边的数据
    quick_sort2(li, left + 1, end)


"""
[5,3,7,6,4,1,0,2,9,10,8]
K=5,<-8 find 2<5 then 5 swap 2    list[l]=5 list[r]=2
[2,3,7,6,4,1,0,5,9,10,8]
 ^             ^
K=5 2-> find 7>5 then 7 swap 5    list[l]=5 list[r]=7
[2,3,5,6,4,1,0,7,9,10,8]
     ^         ^
K=5,<-7 find 0<5 then 5 swap 0    list[l]=0 list[r]=5
[2,3,0,6,4,1,5,7,9,10,8]
     ^       ^
K=5,0-> find 6>5 then 6 swap 5    list[l]=5 list[r]=6
[2,3,0,5,4,1,6,7,9,10,8]
       ^     ^
K=5,<-6 find 1<5 then 5 swap 1    list[l]=1 list[r]=5
[2,3,0,1,4,5,6,7,9,10,8]
       ^   ^
K=5,1-> find 6>5 then 5 swap 1    list[l]=list[r]=5
[2,3,0,1,4,5,6,7,9,10,8]
           ^


[2,3,0,1,4]
K=2 <-4
[1,3,0,2,4]
 ^     ^
K=2 1->
[1,2,0,3,4]
   ^   ^
K=2 <-3
[1,0,2,3,4]
   ^ ^
K=2 0-> list[l]=list[r]=2
[1,0,2,3,4]
     ^


[1,0]
K=1 <-2
[0,1]
 ^ ^  
[0,1] 1-> list[l]=list[r]=1
   ^
   

[6,7,9,10,8]
K=6 <-8
[6,7,9,10,8]
 ^
 
 
[7,9,10,8]
K=7 <-8
[7,9,10,8]
 ^
 

[9,10,8]
K=9 <-8
[8,10,9]
 ^    ^
K=9 8->
[8,9,10]
   ^  ^
K=9 <-10
[8,9,10]
   ^
"""

if __name__ == '__main__':
    # arr = [10, 7, 8, 9, 1, 5]
    arr = [6, 8, 7, 9, 5]
    n = len(arr)
    quick_sort(arr, 0, n - 1)
    print(arr)

    print("------")

    quick_sort2(arr, 0, n - 1)
    print(arr)
