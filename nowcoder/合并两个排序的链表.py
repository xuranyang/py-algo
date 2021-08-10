# 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
# 输入：
# {1,3,5},{2,4,6}
# 复制
# 返回值：
# {1,2,3,4,5,6}

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    """
    递归版本
    """
    # 返回合并后列表
    def Merge(self, pHead1: ListNode, pHead2: ListNode) -> ListNode:
        # write code here
        last = None
        if pHead1 == None:
            return pHead2
        if pHead2 == None:
            return pHead1

        if pHead1.val < pHead2.val:
            last = pHead1
            pHead1.next = self.Merge(pHead1.next, pHead2)
        else:
            last = pHead2
            pHead2.next = self.Merge(pHead1, pHead2.next)

        return last


    """
    迭代版本
    """
    def Merge(self, pHead1: ListNode, pHead2: ListNode) -> ListNode:
        if pHead1 == None:
            return pHead2
        if pHead2 == None:
            return pHead1

        if pHead1.val <= pHead2.val:
            newHead = cur = ListNode(pHead1.val)
            pHead1 = pHead1.next
        else:
            newHead = cur = ListNode(pHead2.val)
            pHead2 = pHead2.next

        while (pHead1 is not None) and (pHead2 is not None):
            if pHead1.val <= pHead2.val:
                cur.next = pHead1
                pHead1 = pHead1.next
            else:
                cur.next = pHead2
                pHead2 = pHead2.next
            cur = cur.next

        if pHead1 is None:
            cur.next = pHead2
        elif pHead2 is None:
            cur.next = pHead1

        return newHead


if __name__ == '__main__':
    p1 = ListNode(1)
    p1.next = ListNode(3)
    p1.next.next = ListNode(5)

    p2 = ListNode(2)
    p2.next = ListNode(4)
    p2.next.next = ListNode(6)

    res = Solution().Merge(p1, p2)
    # debug
    # print(res)

    while res.val:
        print(res.val)
        res = res.next
        if not res:
            break
