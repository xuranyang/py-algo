"""
桶排序
(计数排序的升级版)

桶排序 (Bucket sort)的工作的原理：假设输入数据服从均匀分布，将数据分到有限数量的桶里，
每个桶再分别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排）。

设置一个定量的数组当作空桶；
遍历输入数据，并且把数据一个一个放到对应的桶里去；
对每个不是空的桶进行排序；
从不是空的桶里把排好序的数据拼接起来。
"""


def bucket_sort(arr: list, n):
    # 1.创建n个空桶
    new_list = [[] for _ in range(n)]

    # 每个桶的大小,向上取整
    size = int(max(arr) / n) + 1

    # 2.把arr[i] 插入到bucket[n*arr[i]]
    for data in arr:
        index = int(data / size)
        print(index)
        new_list[index].append(data)

    # 3.桶内排序
    for i in range(n):
        new_list[i].sort()

    # 4.产生新的排序后的列表
    index = 0
    for i in range(n):
        for j in range(len(new_list[i])):
            arr[index] = new_list[i][j]
            index += 1
    return arr


if __name__ == '__main__':
    li = [14, 2, 34, 43, 21, 19]
    buckets = len(li)
    # 每个值一个桶,此时相当于计数排序
    sorted_li = bucket_sort(li, buckets)
    # 分2个桶
    # sorted_li = bucket_sort(li, 2)
    # 分3个桶
    # sorted_li = bucket_sort(li, 3)
    print(sorted_li)
