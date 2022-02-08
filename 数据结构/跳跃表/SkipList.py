import random


class Node(object):
    """
    跳跃表节点
    """

    def __init__(self, key, level):
        self.key = key

        # 当前节点的指向的下一个节点, 用列表维护对应的层数, 列表的索引是层数, 对象是节点
        self.forward = [None] * (level + 1)

    def __str__(self):
        return "Node({})".format(str(self.key))


class SkipList(object):
    """
    跳跃表
    """

    def __init__(self, max_lvl, P):
        self.max_level = max_lvl  # 最高层数
        self.P = P  # 掷硬币的建层概率
        self.header = Node(-1, self.max_level)  # 初始化头节点
        self.level = 0  # 当前层数

    def random_level(self):
        lvl = 0
        while random.random() < self.P and lvl < self.max_level:
            lvl += 1

        return lvl

    def insertElement(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        """
        从跳越列表的最高层开始向后移动当前引用
        当要插入的键大于当前节点旁边的键值时，向后移动当前引用
        否则在 update 中插入当前值，向下移动一层并继续搜索
        """
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]  # 往右挪
            update[i] = current

        current = current.forward[0]

        """
        如果 current 是 NULL 意味着我们已经到达了列表尾部或当前节点和要插入的节点值不一样, 我们要在 update[0] 和 current 之间插入
        """
        if current is None or current.key != key:
            # 为节点随机生成层数
            rlevel = self.random_level()

            # 如果超过当前层, 补全中间层
            if rlevel > self.level:
                for i in range(self.level + 1, rlevel + 1):
                    # update[i]和 self.header是同一个对象,修改update[i]相当于修改self.header
                    update[i] = self.header
                # 更新当前跳跃表的层数
                self.level = rlevel

            # 生成新的节点
            n = Node(key, rlevel)

            # 插入每一层
            for i in range(rlevel + 1):
                n.forward[i] = update[i].forward[i]
                update[i].forward[i] = n

            # print("Successfully inserted key {}".format(key))

    def delete_element(self, search_key):

        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < search_key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is not None and current.key == search_key:

            for i in range(self.level + 1):

                # 如果往上层没有要删除的节点则提前结束
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            del current

            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            print("Successfully deleted {}".format(search_key))

    def search_element(self, key):
        current = self.header
        n = 0
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
                n += 1
        print(n, "次")

        current = current.forward[0]

        if current and current.key == key:
            print("Found key ", key)
            pass

    def display_list(self):
        print("*****Skip List******")
        head = self.header
        for lvl in range(self.level + 1):
            print("Level {}: ".format(lvl))
            node = head.forward[lvl]
            node_list = []
            while node is not None:
                node_list.append(str(node.key))
                node = node.forward[lvl]
            print("->".join(node_list))


if __name__ == '__main__':
    lst = SkipList(5, 0.5)
    lst.insertElement(3)
    lst.insertElement(7)
    lst.insertElement(12)
    lst.insertElement(6)
    lst.insertElement(19)
    lst.insertElement(9)
    lst.insertElement(26)
    lst.insertElement(21)
    lst.insertElement(17)
    lst.insertElement(25)
    lst.display_list()

    print("-" * 100)
    lst.search_element(12)
    print("-" * 100)

    lst.delete_element(12)
    lst.display_list()

    # print("*" * 20)
    # print(lst.header)
    # print(lst.header.forward[0])
    # print(lst.header.forward[0].forward[0])
    # print(lst.header.forward[0].forward[0].forward[0])
    #
    # print("*" * 20)
    # print(lst.header.forward[2])
    # print(lst.header.forward[2].forward[2])
    # print(lst.header.forward[2].forward[2].forward[2])
