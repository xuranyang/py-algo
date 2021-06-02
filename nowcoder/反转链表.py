#!/usr/bin/env python
# -*- coding:utf-8 -*-


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 返回ListNode
    def ReverseList(self, pHead: ListNode):

        if pHead == None or pHead.next == None:
            return pHead

        # 最后一个节点必为None
        pre = None
        # 当前
        cur = pHead

        while cur != None:
            # tmp=2 tmp保存下一个cur
            tmp = cur.next
            # next=None 反转
            cur.next = pre
            # pre = 1 现在的cur为下一个的pre
            pre = cur
            # cur = 2 下一个cur就是之前保存的tmp
            cur = tmp

        return pre


"""
Step0:
     Head    
None   1  ->  2   ->  3  ->  None
Pre   cur

Step1:
            tmp 
None<-  1    2    3    None
       Pre  cur

Step2:
                   tmp 
None <-  1 <-  2    3    None
              Pre  cur


Step3:
                          tmp 
None <-  1 <-  2 <-  3    None
                    Pre   cur

结束
"""
if __name__ == '__main__':
    # 1->2->3->None
    listNode = ListNode(1)
    listNode2 = listNode.next = ListNode(2)
    listNode2.next = ListNode(3)
    print(listNode)

    # 反转以后3->2->1->None
    solution = Solution()
    reverseList = solution.ReverseList(listNode)
    print(reverseList)
