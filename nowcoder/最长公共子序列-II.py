"""
给定两个字符串str1和str2，输出两个字符串的最长公共子序列。
如果最长公共子序列为空，则返回"-1"。
目前给出的数据，仅仅会存在一个最长的公共子序列

输入："1A2C3D4B56","B1D23A456A"
返回值："123456"

输入："abc","def"
返回值："-1"


输入："abc","abc"
返回值："abc"
"""
import numpy
import pprint

"""
状态转移方程:
            0                           i==0 or j==0
dp[i][j]=   dp[i-1][j-1]+1              i,j>0   s1[i]==s2[j]
            max(dp[i-1][j],dp[i][j-1])  i,j>0   s1[i]!=s2[j]
"""


class Solution:
    """
    常规写法
    动态规划写法一
    """
    def LCS(self, s1, s2):
        # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
        m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
        # d用来记录转移方向
        d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

        for p1 in range(len(s1)):
            for p2 in range(len(s2)):
                if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                    m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                    d[p1 + 1][p2 + 1] = 'ok'
                elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                    m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                    d[p1 + 1][p2 + 1] = 'left'
                else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                    m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                    d[p1 + 1][p2 + 1] = 'up'
        (p1, p2) = (len(s1), len(s2))

        # print(numpy.array(d))
        s = []
        while m[p1][p2]:  # 不为None时
            c = d[p1][p2]
            if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
                s.append(s1[p1 - 1])
                p1 -= 1
                p2 -= 1
            if c == 'left':  # 根据标记，向左找下一个
                p2 -= 1
            if c == 'up':  # 根据标记，向上找下一个
                p1 -= 1
        s.reverse()
        pprint.pprint(m)
        pprint.pprint(d)
        if len(s) == 0:
            return "-1"
        return ''.join(s)


    """
    常规写法
    递归实现
    """
    def LCS(self, s1, s2):
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # pprint.pprint(dp)
        i = len(s1) - 1
        j = len(s2) - 1
        res = []
        while i >= 0 and j >= 0:
            if s1[i] == s2[j]:
                res.append(s1[i])
                i -= 1
                j -= 1
            else:
                if dp[i][j + 1] > dp[i + 1][j]:
                    i -= 1
                elif dp[i][j + 1] <= dp[i + 1][j]:
                    j -= 1

        if len(res) > 0:
            return ''.join(reversed(res))
        else:
            return -1


    """
    常规写法
    动态规划写法二
    """
    def LCS(self, s1, s2):
        s1 = ' ' + s1
        s2 = ' ' + s2
        len_s1 = len(s1)
        len_s2 = len(s2)
        dp = [[0] * len_s2 for _ in range(len_s1)]

        for i in range(1, len_s1):
            for j in range(1, len_s2):
                if s1[i] == s2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                elif dp[i - 1][j] > dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                elif dp[i - 1][j] <= dp[i][j - 1]:
                    dp[i][j] = dp[i][j - 1]

        pprint.pprint(dp)

        i = len_s1 - 1
        j = len_s2 - 1

        res = []
        while i > 0 and j > 0:
            if s1[i] == s2[j]:
                res.append(s1[i])
                i -= 1
                j -= 1
            else:
                if dp[i - 1][j] > dp[i][j - 1]:
                    i -= 1
                elif dp[i - 1][j] <= dp[i][j - 1]:
                    j -= 1

        if len(res) > 0:
            return ''.join(reversed(res))
        else:
            return -1


    """
    优化后的取巧写法
    动态规划写法三
    """
    def LCS(self, s1, s2):
        # write code here
        # 动态规划 dp[i][j]表示s1前i个和s2前j个最长公共字符串
        s1 = ' ' + s1
        s2 = ' ' + s2
        n = len(s1)
        m = len(s2)
        dp = [[''] * m for _ in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                if s1[i] == s2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + s1[i]
                else:
                    # 不相等  最长的公共子串应该是dp[i-1][j]和dp[i][j-1]最长的
                    if len(dp[i - 1][j]) > len(dp[i][j - 1]):
                        dp[i][j] = dp[i - 1][j]
                    else:
                        dp[i][j] = dp[i][j - 1]

        # pprint.pprint(dp)

        if len(dp[-1][-1]) == 0:
            return -1
        else:
            return dp[-1][-1]


"""
      B  1  D  2  3  A  4  5  6  A
 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
1 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
A [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
2 [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2],
C [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2],
3 [0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3],
D [0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3],
4 [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4],
B [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
5 [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5],
6 [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6]]


         B    1    D    2    3    A    4    5    6    A
 [[None None None None None None None None None None None]
1 [None 'up' 'ok' 'left' 'left' 'left' 'left' 'left' 'left' 'left' 'left']
A [None 'up' 'up' 'up' 'up' 'up' 'ok' 'left' 'left' 'left' 'ok']
2 [None 'up' 'up' 'up' 'ok' 'left' 'up' 'up' 'up' 'up' 'up']
C [None 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'up']
3 [None 'up' 'up' 'up' 'up' 'ok' 'left' 'left' 'left' 'left' 'left']
D [None 'up' 'up' 'ok' 'up' 'up' 'up' 'up' 'up' 'up' 'up']
4 [None 'up' 'up' 'up' 'up' 'up' 'up' 'ok' 'left' 'left' 'left']
B [None 'ok' 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'up']
5 [None 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'ok' 'left' 'left']
6 [None 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'up' 'ok' 'left']]
"""
if __name__ == '__main__':
    str1 = '1A2C3D4B56'
    str2 = 'B1D23A456A'
    print(Solution().LCS(str1, str2))
