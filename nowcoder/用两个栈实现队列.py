#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
栈A用来作入队列
栈B用来出队列，当栈B为空时，栈A全部出栈到栈B,栈B再出栈（即出队列）
"""
class Solution:
    def __init__(self):
        self.inputStack = []
        self.outputStack = []

    def push(self, node):
        # write code here
        self.inputStack.append(node)

    def pop(self):
        # return xx
        if self.outputStack == []:
            while self.inputStack:
                self.outputStack.append(self.inputStack.pop())

        if self.outputStack:
            return self.outputStack.pop()
        else:
            return None


if __name__ == '__main__':
    solution = Solution()
    solution.push(1)
    solution.push(2)
    solution.push(3)

    print(solution.pop())
    print(solution.pop())

    solution.push(4)
    print(solution.pop())
    solution.push(5)
    print(solution.pop())
    print(solution.pop())
