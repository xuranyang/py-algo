"""
对于一个字符串，请设计一个高效算法，计算其中最长回文子串的长度。
给定字符串A以及它的长度n，请返回最长回文子串的长度。

输入："abc1234321ab",12
输出：7
"""

"""
纯暴力
动态规划
中心扩散法
Manacher算法
"""
import pprint


class Solution:
    """
    纯暴力解法
    """

    def getLongestPalindrome(self, A, n):
        """
        :param A:
        :param n: n=len(A)
        :return:
        """
        # write code here
        max_len, result = float("-inf"), ""
        for i in range(len(A)):
            for j in range(i + 1, len(A) + 1):
                if A[i:j] == A[i:j][::-1]:
                    if j - i > max_len:
                        max_len = j - i
                        result = A[i:j]
        # return result
        return max_len

    """
    动态规划
    令dp[i][j]表示s[i]至s[j]所表示的子串是否是回文子串，是则为1，不是为0
    dp[i][j] = dp[i+1][j-1]   s[i]==s[j]
               0              s[i]!=s[j]
    """

    def getLongestPalindrome(self, A, n):
        """
        :param A: input str
        :type A: str
        :param n: n=len(A)
        :type n: int
        :return:
        :rtype: int
        """
        # write code here
        s = A
        if not s:
            return ""

        s_len = len(s)
        memo = [[0] * s_len for _ in range(s_len)]
        left, right, result_len = 0, 0, 0

        for j in range(s_len):
            for i in range(j):
                if s[i] == s[j] and (j - i < 2 or memo[i + 1][j - 1] == 1):
                    memo[i][j] = 1
                if memo[i][j] == 1 and result_len < j - i + 1:
                    result_len = j - i + 1
                    left, right = i, j
            memo[j][j] = 1

        # pprint.pprint(memo)
        # return s[left:right + 1]
        return result_len

    def getLongestPalindrome(self, A, n):
        s = A
        if not s or len(s) < 1:
            return ""

        def expandAroundCenter(left, right):
            L, R = left, right
            while L >= 0 and R < len(s) and s[L] == s[R]:
                L -= 1
                R += 1
            return R - L - 1

        left, right = 0, 0
        for i in range(len(s)):
            len1 = expandAroundCenter(i, i)  # 奇数
            len2 = expandAroundCenter(i, i + 1)  # 偶数
            max_len = max(len1, len2)
            if max_len > right - left:
                left = i - (max_len - 1) // 2
                right = i + max_len // 2

        # return s[left:right + 1]
        return right - left + 1


if __name__ == '__main__':
    ss = "abc1234321ab"
    solution = Solution()
    res = solution.getLongestPalindrome(ss, len(ss))
    print(res)
