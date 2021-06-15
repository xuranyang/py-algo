#! /usr/bin/env python
# -*- coding: utf-8 -*-

# 输入：1->3->4, 2->5->6
# 输出：1->2->3->4->5->6
"""
题目解析:

1.首先，设定一个虚拟节点 dummy 用来存储结果，循环对比 L1 和 L2 节点上的数字，通过调整 tail 节点的 next 指针来调整 dummy 的结果。

2.如果 L1 当前位置的值小于 L2 ，我们就把  L1 的值接在  tail 节点的后面同时将  L1 指针往后移一个
3.如果 L2 当前位置的值小于等于 L1 ，我们就把  L2 的值接在  tail 节点的后面同时将  L2 指针往后移一个
4.不管我们将哪一个元素接在了 tail 节点后面，都需要向后移一个元素
5.重复以上过程，直到  L1 或者  L2 指向了 null

6.在循环终止的时候，L1 和 L2 至多有一个是非空的。
由于输入的两个链表都是有序的，所以不管哪个链表是非空的，它包含的所有元素都比前面已经合并链表中的所有元素都要大。
这意味着我们只需要简单地将非空链表接在合并链表的后面，并返回合并链表。
"""

"""
https://leetcode-cn.com/problems/merge-two-sorted-lists/solution/he-bing-liang-ge-you-xu-lian-biao-by-leetcode-solu/

1->3->4  2->5->6     dum: 0->None
3->4     2->5->6          0—>1
3->4     5->6             0—>1->2
4->None  5->6             0—>1->2->3
         5->6             0—>1->2->3->4
                          0—>1->2->3->4->5->6

"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# @param l1 ListNode类
# @param l2 ListNode类
# @return ListNode类
#
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # write code here
        # 声明一个头结点，赋初值为0，这个值没有用
        dum = ListNode(0)
        # 不断移动的指针,方便插入
        tail = dum
        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            # else
            elif l1.val >= l2.val:
                tail.next = l2
                l2 = l2.next

            tail = tail.next

        if l1 is not None:
            tail.next = l1

        if l2 is not None:
            tail.next = l2

        return dum.next


def traverse(list_node: ListNode):
    res = ""
    cur = list_node
    while cur is not None:
        res += "%s->" % cur.val
        # print(cur.val)
        cur = cur.next

    return res.strip("->")


if __name__ == '__main__':
    # 1->3->4
    L4 = ListNode(4)
    L3 = ListNode(3)
    L3.next = L4
    L1 = ListNode(1)
    L1.next = L3

    # 2->5->6
    L6 = ListNode(6)
    L5 = ListNode(5)
    L5.next = L6
    L2 = ListNode(2)
    L2.next = L5

    res_obj = Solution().mergeTwoLists(L1, L2)
    # print(res_obj)
    res = traverse(res_obj)
    # 1->2->3->4->5->6
    print(res)
