# 斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
#
#
# F(0) = 0，F(1) = 1
# F(n) = F(n - 1) + F(n - 2)，其中 n > 1
#
#
#  给你 n ，请计算 F(n) 。
#
#
#
#  示例 1：
#
#
# 输入：2
# 输出：1
# 解释：F(2) = F(1) + F(0) = 1 + 0 = 1
#
#
#  示例 2：
#
#
# 输入：3
# 输出：2
# 解释：F(3) = F(2) + F(1) = 1 + 1 = 2
#
#
#  示例 3：
#
#
# 输入：4
# 输出：3
# 解释：F(4) = F(3) + F(2) = 2 + 1 = 3
#
#
#
#
#  提示：
#
#
#  0 <= n <= 30
#
#  Related Topics 数组
#  👍 261 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    """
    暴利递归
    """

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0 or n == 1:
            return n
        else:
            return self.fib(n - 1) + self.fib(n - 2)

    """
    带备忘录的递归解法
    自顶向下
    """

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 1:
            return 0

        memo = [0] * (n + 1)
        # memo = [0 for i in range(n+1)]

        return self.helper(memo, n)

    def helper(self, memo, n):
        """
        :param memo:list
        :param n: int
        :return:
        """
        if n == 1 or n == 2:
            return 1
        if memo[n] != 0:
            return memo[n]

        memo[n] = self.helper(memo, n - 1) + self.helper(memo, n - 2)

        return memo[n]

    """
    dp 数组的迭代解法
    自底向上
    """

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 1:
            return 0
        if n == 1 or n == 2:
            return 1

        dp = [0] * (n + 1)
        # dp = [0 for i in range(n + 1)]
        dp[1] = dp[2] = 1

        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]

    """
    状态压缩
    状态转移方程:
            1               n=1,2
    f(n)=
            f(n-1)+f(n-2)   n>=2
    """

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 1:
            return 0
        if n == 1 or n == 2:
            return 1

        prev = curr = 1

        for i in range(3, n + 1):
            sum = prev + curr
            prev = curr
            curr = sum

        return sum


# leetcode submit region end(Prohibit modification and deletion)


if __name__ == '__main__':
    s = Solution()
    print(str(s.fib(5)))
