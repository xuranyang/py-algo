"""
基数排序
取得数组中的最大数，并取得位数；
arr为原始数组，从最低位开始取每个位组成radix数组；
对radix进行计数排序（利用计数排序适用于小范围数的特点）；

1. 将所有待排序整数（注意，必须是非负整数）统一为位数相同的整数，位数较少的前面补零。
一般用10进制，也可以用16进制甚至2进制。所以前提是能够找到最大值，得到最长的位数，设 k 进制下最长为位数为 d 。
2. 从最低位开始，依次进行一次稳定排序。这样从最低位一直到最高位排序完成以后，整个序列就变成了一个有序序列。
举个例子，有一个整数序列，0, 123, 45, 386, 106，下面是排序过程：

第一次排序，个位，000 123 045 386 106，无任何变化
第二次排序，十位，000 106 123 045 386
第三次排序，百位，000 045 106 123 386
最终结果，0, 45, 106, 123, 386, 排序完成。
"""


def radix_sort(arr: list) -> list:
    """
    10进制基数排序
    :param arr:
    :return:
    """
    current_place = 0  # 记录当前正在排哪一位,初始为个位排序
    max_num = max(arr)  # 最大值
    max_place = len(str(max_num))  # 最大位数
    # 比较规范的写法
    # max_place = 1
    # while max_num >= 10 ** max_place:
    #     max_place += 1
    while current_place < max_place:
        # print("当前arr:" + str(arr))
        bucket_list = [[] for _ in range(10)]
        for num in arr:
            # 将对应的数组元素加入到相应位基数的桶中
            bucket_list[int(num / (10 ** current_place)) % 10].append(num)

        # 查看 bucket_list
        # print("桶数组" + str(bucket_list))

        arr.clear()

        for bucket in bucket_list:
            for num in bucket:
                arr.append(num)
        current_place += 1

    return arr


if __name__ == '__main__':
    li = [21, 14, 5, 2, 1, 34, 43, 19, 202, 101]
    # li = [12, 3, 45, 3543, 214, 1, 4553]
    sorted_li = radix_sort(li)
    print(sorted_li)
