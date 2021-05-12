#
# lru design
# @param operators int整型二维数组 the ops
# @param k int整型 the k
# @return int整型一维数组
#
import sys


class Solution:
    def __init__(self, k):
        """
        :param k: LRU缓存容量
        """
        self.k = k
        self.out = []
        # lst 只存key
        self.lst = []
        # d 保存完成的 key 和 value
        self.d = {}

    def LRU(self, operators, k):
        """
        :param operators: 1表示set,2表示get
        :param k: LRU缓存容量
        :return:
        """
        GET = 1
        SET = 2

        # write code here
        self.k = k
        for o in operators:
            if o[0] == GET:
                self.set(o[1], o[2])
            elif o[0] == SET:
                self.out.append(self.get(o[1]))
        return self.out

    def get(self, key):
        if key in self.d:
            self.lst.remove(key)
            self.lst.append(key)
            return self.d[key]
        return -1

    def set(self, key, value):
        if key in self.d:
            self.lst.remove(key)
            self.lst.append(key)
        else:
            self.lst.append(key)
            self.d[key] = value
        if len(self.lst) > self.k:
            # 最后面的是最新的,最老的在最前面,移除第0个最老的
            old = self.lst.pop(0)
            self.d.pop(old)


if __name__ == '__main__':
    # line = [[1,1,1],[1,2,2],[1,3,2],[2,1],[1,4,4],[2,2]],3
    # for line in sys.stdin.readlines():
    line = input("请输入:")
    line = line.strip()
    line = line.replace(' ', '')
    a = line.split(']],')
    # 容量
    capacity = int(a[1])
    res = []
    s = Solution(capacity)
    for item in a[0][2:].split('],['):
        m = item.split(',')
        if m[0] == '1':
            s.set(int(m[1]), int(m[2]))
        else:
            res.append(s.get(int(m[1])))
    print(str(res).replace(' ', ''))
