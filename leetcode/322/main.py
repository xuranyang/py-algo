# 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回
#  -1。
#
#  你可以认为每种硬币的数量是无限的。
#
#
#
#  示例 1：
#
#
# 输入：coins = [1, 2, 5], amount = 11
# 输出：3
# 解释：11 = 5 + 5 + 1
#
#  示例 2：
#
#
# 输入：coins = [2], amount = 3
# 输出：-1
#
#  示例 3：
#
#
# 输入：coins = [1], amount = 0
# 输出：0
#
#
#  示例 4：
#
#
# 输入：coins = [1], amount = 1
# 输出：1
#
#
#  示例 5：
#
#
# 输入：coins = [1], amount = 2
# 输出：2
#
#
#
#
#  提示：
#
#
#  1 <= coins.length <= 12
#  1 <= coins[i] <= 231 - 1
#  0 <= amount <= 104
#
#  Related Topics 动态规划
#  👍 1175 👎 0

import sys

"""
明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义。
# 初始化 base case
dp[0][0][...] = base
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
"""

"""
1、确定 base case，这个很简单，显然目标金额 amount 为 0 时算法返回 0，因为不需要任何硬币就已经凑出目标金额了。

2、确定「状态」，也就是原问题和子问题中会变化的变量。由于硬币数量无限，硬币的面额也是题目给定的，
只有目标金额会不断地向 base case 靠近，所以唯一的「状态」就是目标金额 amount。

3、确定「选择」，也就是导致「状态」产生变化的行为。目标金额为什么变化呢，因为你在选择硬币，你每选择一枚硬币，就相当于减少了目标金额。
所以说所有硬币的面值，就是你的「选择」。

4、明确 dp 函数/数组的定义。我们这里讲的是自顶向下的解法，所以会有一个递归的 dp 函数，一般来说函数的参数就是状态转移中会变化的量，也就是上面说到的「状态」；
函数的返回值就是题目要求我们计算的量。就本题来说，状态只有一个，即「目标金额」，题目要求我们计算凑出目标金额所需的最少硬币数量。
所以我们可以这样定义 dp 函数：
dp(n) 的定义：输入一个目标金额 n，返回凑出目标金额 n 的最少硬币数量。
"""


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    """
    暴力递归
    自顶向下
    """

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        """
        dp(n) 的定义：输入一个目标金额 n，返回凑出目标金额 n 的最少硬币数量。
        """

        def dp(n):
            # base case
            if n == 0:
                return 0
            if n < 0:
                return -1
            res = float('INF')

            for coin in coins:
                # 子问题
                subproblem = dp(n - coin)

                # 子问题无解，跳过
                if subproblem == -1:
                    continue

                res = min(res, 1 + subproblem)

            if res != float('INF'):
                return res
            else:
                return -1

        return dp(amount)

    """
    带备忘录的递归
    自顶向下
    """

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # 备忘录
        memo = dict()

        def dp(n):
            # 查备忘录，避免重复计算
            if n in memo:
                return memo[n]

            if n == 0:
                return 0

            if n < 0:
                return -1

            res = float('INF')

            for coin in coins:
                subproblem = dp(n - coin)

                if subproblem == -1:
                    continue

                res = min(res, 1 + subproblem)

            # 记入备忘录
            if res != float('INF'):
                memo[n] = res
            else:
                memo[n] = -1

            return memo[n]

        return dp(amount)

    """
    dp 数组的迭代解法
    自底向上
    """

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        # dp数组的定义：当目标金额为i时，至少需要dp[i]枚硬币凑出。
        # 数组大小为 amount + 1，初始值也为 amount + 1
        dp = [amount + 1] * (amount + 1)

        # base case
        dp[0] = 0

        for i in range(len(dp)):
            for coin in coins:
                if i - coin < 0:
                    continue

                dp[i] = min(dp[i], 1 + dp[i - coin])

        if dp[amount] == amount + 1:
            return -1
        else:
            return dp[amount]


# leetcode submit region end(Prohibit modification and deletion)


if __name__ == '__main__':
    pass
