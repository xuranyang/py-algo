"""
给定两个字符串str1和str2,输出两个字符串的最长公共子串
题目保证str1和str2的最长公共子串存在且唯一。

输入："1AB2345CD","12345EF"
返回值："2345"

"""
import pprint

"""
状态转义方程:
            0                   i=0 or j=0
dp[i,j] =   dp[i-1][j-1]+1      s1[i]==s2[j]
            0                   s1[i]!=s2[j]
"""


class Solution:
    def LCS(self, str1, str2):
        # write code here
        dp = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        mmax = 0  # 最长匹配的长度  
        p = 0  # 最长匹配对应在str1中的最后一位  
        for i in range(len(str1)):
            for j in range(len(str2)):
                if str1[i] == str2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    if dp[i + 1][j + 1] > mmax:
                        mmax = dp[i + 1][j + 1]
                        p = i + 1

        # pprint.pprint(dp)
        # return str1[p - mmax:p], mmax  # 返回最长子串及其长度  
        return str1[p - mmax:p]  # 返回最长子串及其长度  


"""
      1  2  3  4  5  E  F
 [[0, 0, 0, 0, 0, 0, 0, 0],
1 [0, 1, 0, 0, 0, 0, 0, 0],
A [0, 0, 0, 0, 0, 0, 0, 0],
B [0, 0, 0, 0, 0, 0, 0, 0],
2 [0, 0, 1, 0, 0, 0, 0, 0],
3 [0, 0, 0, 2, 0, 0, 0, 0],
4 [0, 0, 0, 0, 3, 0, 0, 0],
5 [0, 0, 0, 0, 0, 4, 0, 0],
C [0, 0, 0, 0, 0, 0, 0, 0],
D [0, 0, 0, 0, 0, 0, 0, 0]]
"""
if __name__ == '__main__':
    s1 = "1AB2345CD"
    s2 = "12345EF"

    print(Solution().LCS(s1, s2))
