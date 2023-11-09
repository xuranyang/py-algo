import random

"""
在向跳跃表中插入新的结点时候，我们需要生成该结点的层数。
使用抛硬币的思想随机生成层数，如果是正面（random.randint(0, 1) == 1）则层数加一，直到抛出反面为止。
其中的 MAX_DEPTH 是防止如果运气太好，层数就会太高，而太高的层数往往并不会提供额外的性能。
"""
MAX_DEPTH = 5


class SkipNode:
    def __init__(self, height=0, elem=None):
        self.elem = elem
        self.next = [None] * height

    def __repr__(self):
        return str(self.elem)


class SkipList:
    def __init__(self):
        self.head = SkipNode()

    def updateList(self, elem):
        """
        从跳跃表的最顶层开始依次向下查找，
        找到该层级中比给定元素elem小的最大一个元素，将该元素保存起来，
        重复以上步骤知道到达最底层
        它返回一个列表update，update[0]表示第一层最后一个比elem小的元素，以此类推。该方法可以使得插入删除操作变得更加简单。
        """
        update = [None] * len(self.head.next)
        current = self.head

        for i in reversed(range(len(self.head.next))):
            while current.next[i] != None and current.next[i].elem < elem:
                current = current.next[i]
            # update[i]=current=self.head,即 update[i] 和 self.head 是同一个对象,修改update[i]相当于修改self.head
            update[i] = current

        return update

    def find(self, elem, update=None):
        if update == None:
            # 获得每一层比待查找元素elem小的最大元素列表
            update = self.updateList(elem)
        if len(update) > 0:
            # update[0]为第一层最后一个比elem小的元素
            # update[0].next[0] 为 update[0]的下一个元素,
            # 如果待查找元素存在的话,update[0]的下一个元素应该就等于elem
            # 即 update[0].next[0]==elem,如果不相等就说明待查找元素不存在
            candidate = update[0].next[0]
            if candidate != None and candidate.elem == elem:
                return candidate
        return None

    def insert(self, insert_elem):
        node = SkipNode(self.randomHeight(), insert_elem)

        while len(self.head.next) < len(node.next):
            self.head.next.append(None)

        update = self.updateList(insert_elem)
        # 查找insert_elem是否已经在跳跃表中存在,如果不存在就插入每一层
        if self.find(insert_elem, update) == None:
            # len(node.next) 跳表的层数,遍历插入每一层
            for i in range(len(node.next)):
                node.next[i] = update[i].next[i]
                update[i].next[i] = node

    def randomHeight(self):
        k = 1
        while random.randint(0, 1):
            k = k + 1
            if k == MAX_DEPTH:
                break
        return k

    def remove(self, elem):
        update = self.updateList(elem)
        find = self.find(elem, update)
        if find != None:
            for i in range(len(find.next)):
                update[i].next[i] = find.next[i]
                if self.head.next[i] == None:
                    self.head.next = self.head.next[:i]
                    return

    def traversal(self):
        for i in reversed(range(len(self.head.next))):
            head = self.head
            line = []
            while head.next[i] != None:
                line.append(str(head.next[i].elem))
                head = head.next[i]
            print('line{}: '.format(i + 1) + '->'.join(line))


if __name__ == '__main__':
    skip_list = SkipList()
    skip_list.insert(3)
    skip_list.insert(7)
    skip_list.insert(12)
    skip_list.insert(6)
    skip_list.insert(19)
    skip_list.insert(9)
    skip_list.insert(26)
    skip_list.insert(21)
    skip_list.insert(17)
    skip_list.insert(25)

    skip_list.traversal()
