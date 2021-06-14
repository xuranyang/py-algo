# 有一个整数数组，请你根据快速排序的思路，找出数组中第K大的数。
# 给定一个整数数组a,同时给定它的大小n和要找的K(K在1到n之间)，请返回第K大的数，保证答案存在。
# 输入：[1,3,5,2,2],5,3
# 返回值：2


"""
可以结合快排一起看
在 前大后小 快排的基础上,增加了一个条件判断
"""
class Solution:
    def findKth(self, a, n, K):
        # write code here
        return self.quick_sort(a, 0, n - 1, K)

    def quick_sort(self, arr, left, right, k):
        if left <= right:
            index = self.partition(arr, left, right)

            # 因为index从0开始,所以k-1位置就是第k大的数
            # 如果正好index就是第k大的数,该arr[index]就是结果
            if index == k - 1:
                return arr[index]
            # 此时,说明第k大的数比arr[index]更小
            # 第k大的数,在arr中的位置一定是在 index+1及之后
            elif index < k - 1:
                return self.quick_sort(arr, index + 1, right, k)
            # 反之,index>=k-1,说明第k大的数 比 arr[index]更大
            # 第k大的数,在arr中的位置一定是在 index-1及之前
            else:
                return self.quick_sort(arr, left, index - 1, k)

    # 根据 privot 前大后小 进行分区,
    # 大于privot的前一个分区，
    # 小于等于privot的后一个分区
    # 第一次初始化时,left为0,返回值是 根据arr[0]进行分割的index
    # 保证index之前都大于arr[0],index之后都小于等于arr[0]
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
