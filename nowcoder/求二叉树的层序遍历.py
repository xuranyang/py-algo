class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# @param root TreeNode类
# @return int整型二维数组

class Solution:
    # 广度优先
    def levelOrder(self, root):
        # write code here
        if root is None:
            return []
        # 初始化队列
        queue = [root]
        # 初始化最终层序遍历 结果列表
        res = []
        while len(queue) > 0:
            # 初始化每一层 val 的list
            level_res = []
            for _ in range(len(queue)):
                # 遍历取出当前层的每一个节点
                cur = queue.pop(0)
                # 取出当前层节点的val,放入该层的list
                level_res.append(cur.val)
                # 下一层的左节点
                if cur.left is not None:
                    queue.append(cur.left)
                # 下一层的右节点
                if cur.right is not None:
                    queue.append(cur.right)
            # 放入最终结果的列表中
            res.append(level_res)

        return res


"""
   3
9     20
    15  7

返回结果:
[
    [3],
    [9,20],
    [15,7]
]
"""
if __name__ == '__main__':
    tree = TreeNode(3)
    tree.left = TreeNode(9)
    tree_son = TreeNode(20)
    tree_son.left = TreeNode(15)
    tree_son.right = TreeNode(7)
    tree.right = tree_son

    level_order_res = Solution().levelOrder(tree)
    print(level_order_res)
