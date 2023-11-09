from bitarray import bitarray
import mmh3

"""
https://www.cnblogs.com/naive/p/5815433.html
https://blog.csdn.net/qq_37674086/article/details/111396241

错误率估计为(1-exp(-kn/m))^k
https://www.cnblogs.com/liyulong1982/p/6013002.html
https://www.zhihu.com/question/38573286/answer/507497251

将要查询的元素给k个哈希函数
得到对应于位数组上的k个位置
布隆过滤器的判断结果只有以下两种：
*肯定不存在：如果k个位置有一个为0，则肯定不在集合中
*可能存在：如果k个位置全部为1，则可能在集合中


"""

class BloomFilter(set):

    def __init__(self, size, hash_count):
        super(BloomFilter, self).__init__()
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.size = size
        self.hash_count = hash_count

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.bit_array)

    def add(self, item):
        for ii in range(self.hash_count):
            index = mmh3.hash(item, ii) % self.size
            self.bit_array[index] = 1

        return self

    def __contains__(self, item):
        out = True
        for ii in range(self.hash_count):
            index = mmh3.hash(item, ii) % self.size
            if self.bit_array[index] == 0:
                out = False

        return out


def main():
    # bloom = BloomFilter(1000, 10)
    # bloom = BloomFilter(100, 1)
    # bloom = BloomFilter(100, 100)
    bloom = BloomFilter(100, 10)
    animals = ['dog', 'cat', 'giraffe', 'fly', 'mosquito', 'horse', 'eagle',
               'bird', 'bison', 'boar', 'butterfly', 'ant', 'anaconda', 'bear',
               'chicken', 'dolphin', 'donkey', 'crow', 'crocodile']
    # 将animals插入进布隆过滤器
    for animal in animals:
        bloom.add(animal)

    # 在布隆过滤器中的animals
    # animals里的每个animal应该都会被布隆过滤器判断成存在,不应该有任何错误的否定false negatives
    # 如果被布隆过滤器判断成一定不存在,说明一定出现了问题
    for animal in animals:
        if animal in bloom:
            print('[IN]{}符合预期在布隆过滤器中'.format(animal))
        else:
            print('Something is terribly went wrong for {}'.format(animal))
            print('FALSE NEGATIVE!')

    # 不在布隆过滤器中的animals
    # 可能会出现误判 false positives
    other_animals = ['badger', 'cow', 'pig', 'sheep', 'bee', 'wolf', 'fox',
                     'whale', 'shark', 'fish', 'turkey', 'duck', 'dove',
                     'deer', 'elephant', 'frog', 'falcon', 'goat', 'gorilla',
                     'hawk' ]
    for other_animal in other_animals:
        if other_animal in bloom:
            print('[False Positive]{}其实不在布隆过滤器中,但是出现了误判'.format(other_animal))
        else:
            print('[NOT IN]{}符合预期不在布隆过滤器中'.format(other_animal))


if __name__ == '__main__':
    main()
