#! /usr/bin/env python
# -*- coding: utf-8 -*-

class TreeNode(object):
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None

    def insertLeft(self, left_value):
        self.left = TreeNode(left_value)
        return self.left

    def insertRight(self, right_value):
        self.right = TreeNode(right_value)
        return self.right

    def show(self):
        # if self.data is not None:
        #     print(self.data)
        print(self.data)


def preOrder(node):
    """
    先序遍历
    根 左 右
    """
    if node.data is not None:
        # root
        node.show()
        # left
        if node.left is not None:
            preOrder(node.left)
        # right
        if node.right is not None:
            preOrder(node.right)


def inOrder(node):
    """
    中序遍历
    左 根 右
    """
    if node.data is not None:
        # left
        if node.left is not None:
            inOrder(node.left)
        # root
        node.show()
        # right
        if node.right is not None:
            inOrder(node.right)


def postOrder(node):
    """
    后序遍历
    左 右 根
    """
    if node.data is not None:
        # left
        if node.left is not None:
            postOrder(node.left)
        # right
        if node.right is not None:
            postOrder(node.right)
        # root
        node.show()


def BFS(node: TreeNode):
    """
    广度优先
    一层一层遍历
    """
    if node.data is not None:
        # queue队列
        queue = [node]

        while len(queue) > 0:
            # 拿出队首节点
            currentNode = queue.pop(0)
            currentNode.show()
            if currentNode.left is not None:
                queue.append(currentNode.left)
            if currentNode.right is not None:
                queue.append(currentNode.right)


def DFS_Pre(node: TreeNode):
    """
    深度优先(先序遍历、中序遍历、后序遍历)
    这里以先序遍历为例
    """
    if node.data is not None:
        # 栈用来保存未访问的节点
        stack = [node]
        while len(stack) > 0:
            # 拿出栈顶节点
            # 栈先进后出,优先拿出后进的节点,默认为pop(-1)移出最后一个元素
            """
            从下往上看 先出左 再出右
            根 左 右 先序列遍历
            """
            currentNode = stack.pop(-1)
            currentNode.show()

            if currentNode.right is not None:
                stack.append(currentNode.right)

            if currentNode.left is not None:
                stack.append(currentNode.left)


def DFS_In(node: TreeNode):
    """
    深度优先(中序遍历)
    """
    # 栈用来保存未访问的节点
    stack = []
    while len(stack) > 0 or node is not None:
        # 拿出栈顶节点
        # 栈先进后出,优先拿出后进的节点,默认为pop(-1)移出最后一个元素
        """
        左 根 右
        """
        # 先往左边一直遍历，直到最左下角的元素位置
        # 然后遍历左边每个节点右边的元素
        # 左子树遍历完以后 开始遍历右子树
        # 和左子树一样的方法 全部遍历完毕
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop(-1)
        node.show()
        node = node.right


def DFS_Post(node: TreeNode):
    """
    深度优先(后序遍历)
    左 右 根
    """
    stack = []
    while len(stack) > 0 or node is not None:
        # 下行循环，直到找到第一个叶子节点
        while node is not None:
            stack.append(node)
            # 能左就左，不能左就右
            if node.left is not None:
                node = node.left
            else:
                node = node.right


        currentNode = stack.pop(-1)
        currentNode.show()
        # 如果当前节点是上一节点的左子节点，则遍历右子节点
        if len(stack) > 0 and currentNode == stack[-1].left:
            node = stack[-1].right
        else:
            node = None
# root A C
# root A              C
# root A D F          C
# root A D           C F
# root A D G          C F
# root A D          C F G


if __name__ == '__main__':
    """
           root
        None  None
    """
    root = TreeNode("root")
    """
            root
          A      None
      None None
    """
    A = root.insertLeft("A")
    """
             root
          A         B
      None None None None
    """
    B = root.insertRight("B")
    """
                  root
             A           B
         C      None None None
      None None   
    """
    A.insertLeft("C")
    """
                      root
               A              B
         C          D      None None
      None None  None None 
    """
    D = A.insertRight("D")
    """
                      root
               A                 B
         C            D      None None
      None None    F     None
               None None 
    """
    D.insertLeft("F")
    """
                      root
               A                     B
         C              D        None None
      None None    F         G
               None None None None
    """
    D.insertRight("G")
    """
                      root
               A                       B
         C              D        None     E
      None None    F         G        None None
               None None None None
    """
    B.insertRight("E")

    print("先序遍历:")
    # root A C D F G B E
    preOrder(root)
    print("中序遍历:")
    # C A F D G root B E
    inOrder(root)
    print("后序遍历:")
    # C F G D A E B root
    postOrder(root)
    print("广度优先:")
    # root A B C D E F G
    BFS(root)
    print("深度优先:")
    # DFS_Pre(root)
    # DFS_In(root)
    DFS_Post(root)
