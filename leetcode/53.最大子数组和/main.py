"""
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组是数组中的一个连续部分。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

示例 2：
输入：nums = [1]
输出：1

示例 3：
输入：nums = [5,4,-1,7,8]
输出：23
"""
from typing import List

"""
定义状态（定义子问题）:
dp[i]：表示以 nums[i] 结尾 的 连续 子数组的最大和

状态转移方程：
dp[i] = dp[i-1]+nums[i] , if dp[i-1]>0
dp[i] = nums[i] , if dp[i-1]<=0

即：
dp[i] = max( dp[i-1]+nums[i] , nums[i] )

dp[0] = nums[0]
"""


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        size = len(nums)
        if size == 0:
            return 0
        dp = [0 for _ in range(size)]

        dp[0] = nums[0]
        for i in range(1, size):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    def maxSubArray(self, nums: List[int]) -> int:
        size = len(nums)
        pre = 0
        res = nums[0]
        for i in range(size):
            pre = max(nums[i], pre + nums[i])
            res = max(res, pre)
        return res


if __name__ == '__main__':
    # num_list = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    num_list = [5, 4, -1, 7, 8]
    print(Solution().maxSubArray(num_list))
