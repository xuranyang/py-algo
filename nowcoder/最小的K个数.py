#!/usr/bin/env python
# -*- coding:utf-8 -*-
import heapq


class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k == 0 or k > len(tinput):
            return []

        reverse_tinput = [-x for x in tinput[:k]]
        heapq.heapify(reverse_tinput)

        for i in range(k, len(tinput)):
            if -tinput[i] > reverse_tinput[0]:
                heapq.heappop(reverse_tinput)
                heapq.heappush(reverse_tinput, -tinput[i])

        return sorted([-x for x in reverse_tinput])


"""
方法二 手动实现最大堆
"""


def adjustHeap(arr: list, start: int, end: int):
    """
    将位置为 start 的节点,不断与其子节点比较,如果子节点
    :param arr:
    :param start:parent父节点
    :param end:arr的 前end个数
    :return: 返回 以位置为 start 为根节点时,start为该节点递归往下最大的元素
    """
    # 临时存储父节点的值,用于最后的赋值
    tmp = arr[start]
    # 左子节点位置
    child = start * 2 + 1
    while child <= end:
        # 右子节点 为前end个 且 右子节点的值>左子节点的值
        if child + 1 <= end and arr[child + 1] > arr[child]:
            child += 1
        # 如果子节点小于父节点,跳出循环
        if arr[child] < tmp:
            break
        # 只有子节点 大于等于 父节点时,进行后续的交换操作
        # 将父节点的值修改为 值更大的子节点
        arr[start] = arr[child]
        # 将父节点修改为 当前的子节点(新父节点)
        start = child
        # 新父节点的 新左子节点的位置
        child = child * 2 + 1
    # 将新父节点的值修改为 最原始父节点的值
    # (如果没有新父节点,相当于最原始父节点与自己交换,等于没变)
    arr[start] = tmp


def GetLeastNumbers_Solution2(tinput: list, k: int):
    while tinput is None or k <= 0 or k > len(tinput):
        return []

    numbers = tinput[:k]

    # i 初始化为 第k个节点的 父节点, 然后 依次往前一个节点
    # k从1开始计算,index位置从0开始计算,所以要k-1
    for i in range(int(k / 2) - 1, -1, -1):
        adjustHeap(numbers, i, k - 1)

    # 保证前k已经构成最大堆以后,
    # 每次只需要将arr数组中 第k个以后且 小于头节点的 元素与 最大堆的堆顶元素进行替换，
    # 然后最大堆 再重新交换 所有元素顺序
    for i in range(k, len(tinput)):
        if tinput[i] < numbers[0]:
            numbers[0] = tinput[i]
            adjustHeap(numbers, 0, k - 1)

    return sorted(numbers)


"""
[4, 5, 1, 6, 2, 7, 3, 8] 取最小的4个数
[4, 5, 1, 6]
--------------------------------------
Step1:前4个调整顺序
tmp=4
[4, 6, 1, 5]
[6, 6*, 1, 5]
[6, 5, 1, 5*]
[6, 5, 1, 4]

Step2:加入比堆顶6小的元素2
tmp=2
[2, 5, 1, 4]
[5, 5*, 1, 4]
[5, 4, 1, 4*]
[5, 4, 1, 2]

Step3:加入比堆顶5小的元素3
tmp=3
[3, 4, 1, 2]
[4, 4*, 1, 2]
[4, 3, 1, 2]
"""
if __name__ == '__main__':
    arr = [4, 5, 1, 6, 2, 7, 3, 8]
    top_k = 4

    # solution = Solution()
    # res = solution.GetLeastNumbers_Solution(arr, top_k)
    # print(res)

    print(GetLeastNumbers_Solution2(arr, top_k))

    # debug
    # n = arr[:4]
    # for i in range(int(4 / 2) - 1, -1, -1):
    #     adjustHeap(n, i, 4 - 1)
    #     print("%s:%s" % (i, n))
    # for i in range(4, len(arr)):
    #     if arr[i] < n[0]:
    #         n[0] = arr[i]
    #         adjustHeap(n, 0, 4 - 1)
    #         print(n)
