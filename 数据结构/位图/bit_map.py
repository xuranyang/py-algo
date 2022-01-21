"""
假设有10亿个int类型不重复非负数的数字，而我们只有2G的内存空间，如何将其排序？

int类型数据占据4个字节(Byte),1个字节(Byte)占据8个bit位
10亿个int 大约等于 4GB内存
如果使用1bit来存储一个int,只需要原来的1/32的空间,约125MB
python 中的int 是有符号的，所以只支持31个数

https://blog.csdn.net/xc_zhou/article/details/110672513
https://blog.csdn.net/s15885823584/article/details/84894507
https://www.cnblogs.com/cjsblog/p/11613708.html

[input data]:
nums=[n1,n2,n3,...]
max_num = max(nums)

[init arr]:
len(arr)= [ max_num + (32-1) ] / 32
arr = [ 0, 0, 0 ... ]

[set]:
arr[i//32] = arr[i//32] | (1 << i%32 )

[clean]:
arr[i//32] = arr[i//32] & ( ~( 1 << i%32 ) )

[find]:
arr[i//32] = arr[i//32] & (1 << i%32 )
"""


class BitMap:
    def __init__(self, max_value):
        self._size = int((max_value + 31 - 1) / 31)  # 向上取正  确定数组大小
        # self._size = num / 31  # 向下取正  确定数组大小
        self.array = [0 for _ in range(self._size)]  # 初始化为0

    def getElemIndex(self, num):  # 获取该数的数组下标
        return num // 31

    def getBitIndex(self, num):  # 获取该数所在数组的位下标
        return num % 31

    def set(self, num):  # 将该数所在的位置1
        elemIndex = self.getElemIndex(num)
        bitIndex = self.getBitIndex(num)
        self.array[elemIndex] = self.array[elemIndex] | (1 << bitIndex)

    def clean(self, num):  # 将该数所在的位置0
        elemIndex = self.getElemIndex(num)
        bitIndex = self.getBitIndex(num)
        self.array[elemIndex] = self.array[elemIndex] & (~(1 << bitIndex))

    def find(self, num):  # 查找该数是否存在
        elemIndex = self.getElemIndex(num)
        bitIndex = self.getBitIndex(num)
        if self.array[elemIndex] & (1 << bitIndex):
            return True
        return False


if __name__ == '__main__':
    array_list = [45, 2, 78, 35, 67, 90, 879, 0, 340, 123, 46]
    results = []
    bitmap = BitMap(max_value=max(array_list))
    for num in array_list:
        bitmap.set(num)

    for i in range(max(array_list) + 1):
        if bitmap.find(i):
            results.append(i)

    print(bitmap)  # debug
    print(bitmap.array)  # bitmap detail
    print(array_list)
    print(results)
