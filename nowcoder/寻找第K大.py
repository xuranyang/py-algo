# 有一个整数数组，请你根据快速排序的思路，找出数组中第K大的数。
# 给定一个整数数组a,同时给定它的大小n和要找的K(K在1到n之间)，请返回第K大的数，保证答案存在。
# 输入：[1,3,5,2,2],5,3
# 返回值：2

class Solution:
    def findKth(self, a, n, K):
        # write code here
        return self.quick_sort(a, 0, n - 1, K)

    def quick_sort(self, arr, left, right, k):
        if left <= right:
            index = self.partition(arr, left, right)

            if index == k - 1:
                return arr[index]
            elif index < k - 1:
                return self.quick_sort(arr, index + 1, right, k)
            else:
                return self.quick_sort(arr, left, index - 1, k)

    def partition(self, arr, left, right):
        privot = arr[left]
        while left < right:
            while left < right and arr[right] <= privot:
                right -= 1
            arr[left] = arr[right]

            # 也可以取 >=
            while left < right and arr[left] > privot:
                left += 1
            arr[right] = arr[left]

        arr[left] = privot

        return left
