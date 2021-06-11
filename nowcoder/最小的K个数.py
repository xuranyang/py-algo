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
    tmp = arr[start]
    child = start * 2 + 1
    while child <= end:
        if child + 1 <= end and arr[child + 1] > arr[child]:
            child += 1
        if arr[child] < tmp:
            break
        arr[start] = arr[child]
        start = child
        child = child * 2 + 1

    arr[start] = tmp


def GetLeastNumbers_Solution2(tinput: list, k: int):
    while tinput is None or k <= 0 or k > len(tinput):
        return []

    numbers = tinput[:k]

    for i in range(int(k / 2) - 1, -1, -1):
        adjustHeap(numbers, i, k - 1)

    for i in range(k, len(tinput)):
        if tinput[i] < numbers[0]:
            numbers[0] = tinput[i]
            adjustHeap(numbers, 0, k - 1)

    return sorted(numbers)


if __name__ == '__main__':
    arr = [4, 5, 1, 6, 2, 7, 3, 8]
    top_k = 4

    # solution = Solution()
    # res = solution.GetLeastNumbers_Solution(arr, top_k)
    # print(res)

    print(GetLeastNumbers_Solution2(arr, top_k))
