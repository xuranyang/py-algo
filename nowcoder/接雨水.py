"""
输入：[3,1,2,5,2,4]
返回值：5
说明：数组 [3,1,2,5,2,4] 表示柱子高度图，在这种情况下，可以接 5个单位的雨水，蓝色的为雨水
"""
from typing import List

input_list = [3, 1, 2, 5, 2, 4]
# input_list = [4, 5, 1, 3, 2]
# input_list = [3, 1, 2, 5, 2, 4, 3, 3, 4]
"""
      |
      | W |
| W W | W |
| W | | | |
| | | | | |
"""


class Solution:
    """
    双指针
    分别从两边往中间靠，每次都靠短的那边
    """

    def maxWater(self, arr: List[int]) -> int:
        # 排除空数组
        if len(arr) == 0:
            return 0
        res = 0
        # 左右双指针
        left = 0
        right = len(arr) - 1
        # 中间区域的边界高度
        maxL = 0
        maxR = 0
        # 直到左右指针相遇
        while left < right:
            # 每次维护往中间的最大边界
            maxL = max(maxL, arr[left])
            maxR = max(maxR, arr[right])
            # 较短的边界确定该格子的水量
            if maxR > maxL:
                # 左边短
                res += maxL - arr[left]
                left += 1
            else:
                # 右边短
                res += maxR - arr[right]
                right -= 1
        return res

    def max_water(self, arr: list) -> int:
        """
        :param arr:
        :return:
        """
        if len(arr) < 3:
            return 0

        tmp_left_res = []
        tmp_right_res = []  # 最后要 reverse 反转

        left_idx = 0
        right_idx = len(arr) - 1
        max_left_h = arr[left_idx]
        max_right_h = arr[right_idx]

        while left_idx <= right_idx:
            max_left_h = max(max_left_h, arr[left_idx])
            max_right_h = max(max_right_h, arr[right_idx])

            if max_left_h <= max_right_h:
                # 左边短,从左边开始
                tmp_left_res.append(max_left_h)
                left_idx += 1
            else:
                # 右边短,从右边开始
                tmp_right_res.append(max_right_h)
                right_idx -= 1

        tmp_right_res.reverse()
        tmp_res_list = tmp_left_res + tmp_right_res

        res = 0
        for i in range(len(arr)):
            res += tmp_res_list[i] - arr[i]
        return res


if __name__ == '__main__':
    print(Solution().maxWater(input_list))
    print(Solution().max_water(input_list))
