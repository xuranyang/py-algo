#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
"""
import time

"""
把台阶当成一个平面，
也可以反向看，等价于 从第n个台阶往下跳到第0阶的跳法数：
假设f[n]表示剩下n个台阶时的所有跳法数
f[0]=f[1]=1
f[2]=2

当n>=2时，第一次跳的时候有两种不同的选择：
    1.第一次跳1级，此时跳法的数目等于后面剩下的n-1级台阶的跳法数目，即为f(n-1);
    2.第一次跳2级，此时跳法数目等于后面剩下的n-2级台阶的跳法数目，即为f(n-2)；
f[n]=f[n-1]+f[n-2]
"""


class Solution:
    """
    递归
    """
    def jumpFloor(self, number):
        # write code here
        if number <= 1:
            return 1
        else:
            return self.jumpFloor(number - 1) + self.jumpFloor(number - 2)

    """
    自顶向下 动态规划
    """
    def __init__(self):
        self.memo = {}

    def jumpFloor(self, number):
        if number <= 1:
            self.memo[number] = 1
            return 1
        else:

            if number - 1 in self.memo:
                left = self.memo[number - 1]
            else:
                left = self.jumpFloor(number - 1)

            if number - 2 in self.memo:
                right = self.memo[number - 2]
            else:
                right = self.jumpFloor(number - 2)

            self.memo[number] = left + right
            return left + right

    """
    自底向上 动态规划
    """
    def jumpFloor(self, number):
        if number < 2:
            return number

        dp_dict = {}
        dp_dict[0] = dp_dict[1] = 1

        for i in range(2, number + 1):
            dp_dict[i] = dp_dict[i - 1] + dp_dict[i - 2]

        return dp_dict[number]


if __name__ == '__main__':
    start = time.time()
    print(Solution().jumpFloor(34))
    end = time.time()
    print(end - start)
