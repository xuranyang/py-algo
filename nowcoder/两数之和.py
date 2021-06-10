#! /usr/bin/env python
# -*- coding: utf-8 -*-

class Solution:
    def twoSum(self, numbers, target):
        # write code here
        d = {}
        for i in range(len(numbers)):
            if target - numbers[i] in d:
                return [d[target - numbers[i]], i + 1]
            d[numbers[i]] = i + 1


if __name__ == '__main__':
    solution = Solution()
    res = solution.twoSum([3, 2, 4], 6)
    # res = solution.twoSum([0, 4, 3, 0], 0)
    print(res)
