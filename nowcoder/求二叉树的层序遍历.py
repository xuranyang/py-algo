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

        queue = [root]

        res = []
        while len(queue) > 0:
            level_res = []
            for i in range(len(queue)):
                # 遍历取出当前层的每一个节点
                cur = queue.pop(0)
                level_res.append(cur.val)
                # 下一层的左节点
                if cur.left is not None:
                    queue.append(cur.left)
                # 下一层的右节点
                if cur.right is not None:
                    queue.append(cur.right)
            res.append(level_res)

        return res
