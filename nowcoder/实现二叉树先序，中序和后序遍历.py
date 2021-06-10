#! /usr/bin/env python
# -*- coding: utf-8 -*-

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def threeOrders(self, root):
        pre_order, in_order, post_order = [], [], []

        def dfs(root):
            if root is None:
                return None
            # 根 左 右
            pre_order.append(root.val)
            dfs(root.left)
            # 左 根 右
            in_order.append(root.val)
            dfs(root.right)
            # 左 右 根
            post_order.append(root.val)

        dfs(root)

        return [pre_order, in_order, post_order]
