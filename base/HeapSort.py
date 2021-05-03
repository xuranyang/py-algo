import random

"""
堆排序

大顶堆（从小到大排序） 或 小顶堆（从大到小排序）
大顶堆：一棵完全二叉树,满足任一节点都比其孩子节点大
小顶堆：一棵完全二叉树,满足任一节点都比其孩子节点小

假设要排序的数组长度为n,从小到大排序
1.建立大顶堆
2.把堆顶元素和数组最后一个元素交换
3.对数组的前 n-1 个元素重新构造堆（大顶堆或小顶堆）,把堆顶元素和数组最后一个元素交换
4.以此类推,直到数据排序完成
"""

"""
           0
      1        2
    3   4    5   6
   7 8 9 10


完全⼆叉树有个特性：
左边⼦节点位置 = 当前⽗节点的两倍 + 1
右边⼦节点位置 = 当前⽗节点的两倍 + 2
"""

"""
大顶堆建立

 0   1   2   3   4   5
[14, 2, 34, 43, 21, 19]

Step1:begin=2,tmp=34
     14
  2     34
43 21 19


setp2:begin=1,tmp=2
       34
  2(i)      34
43(j) 21  19

       14
   43       34
43(i) 21  19

       14
   43       34
2(tmp) 21  19


step3:begin=0,tmp=14
      14(i)
  43(j)     34
2   21   19


      43
  43(i)     34
2   21(j)   19



      43
  21     34
2   21(i)   19


       43
  21        34
2   14(tmp)   19
结束
大顶堆构造完成:
     43
  21     34
2  14 19
"""


def sift(arr: list, begin: int, end: int):
    """
    :param arr:
    :param begin: 根节点位置
    :param end: 最后一个元素位置
    :return:
    """
    i = begin  # 最开始跟节点的位置
    j = 2 * i + 1  # 左边下一层孩子节点
    tmp = arr[i]  # 把堆顶元素存下来

    while j <= end:  # 只要j位置有节点，有数字便可以一直循环
        if j < end and arr[j + 1] > arr[j]:  # 右边孩子有并且右边更大
            j += 1  # 把j指向j+1，右边孩子大于左边，指向右边

        if arr[j] > tmp:  # 如果孩子节点比tmp更大
            arr[i] = arr[j]  # 父节点的值修改成孩子节点的值
            i = j  # 往下看一层
            j = 2 * i + 1
        else:  # tmp更大的情况，把tmp放上来
            # arr[i] = tmp  # 把tmp放到某一级领导的位置上
            break

    arr[i] = tmp  # 把tmp放在叶子节点上去


"""
     43
  21     34
2  14 19

==== 43
[19, 21, 34, 2, 14, 43]
     19
  21     34
2  14 

[34, 21, 19, 2, 14, 43]
     34
  21     19
2  14 

====  34  43 
[14, 21, 19, 2, 34, 43]
     14
  21     19
2   

[21, 14, 19, 2, 34, 43]
     21
  14     19
2   

====  21 34 43  
[2, 14, 19, 21, 34, 43]
     2
  14   19


[19, 14, 2, 21, 34, 43]
     19
  14    2 

==== 19 21 34 43 
[14, 2, 19, 21, 34, 43]
     14
        2

==== 14 19 21 34 43 
[2, 14, 19, 21, 34, 43]
2

==== 2 14 19 21 34 43
"""


def heap_sort(arr: list):
    # 构造堆
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift(arr, i, n - 1)
    # 以上表示大根堆构造完成
    for i in range(n - 1, -1, -1):  # i是指当前堆的最后一个元素
        arr[0], arr[i] = arr[i], arr[0]  # 根与最后一个元素交换
        sift(arr, 0, i - 1)  # i-1是新的堆的end,对arr[0]到arr[i-1]重新建堆
    # 以上表示吐出来数字的过程
    return arr


"""
堆排序 写法二
"""


def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换

        heapify(arr, n, largest)


def heapSort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n, -1, -1):
        heapify(arr, n, i)

        # 一个个交换元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换
        heapify(arr, i, 0)

    return arr


if __name__ == '__main__':
    li = [14, 2, 34, 43, 21, 19]
    # sorted_li = heap_sort(li)
    # print(sorted_li)
    sorted_li = heapSort(li)
    print(sorted_li)
