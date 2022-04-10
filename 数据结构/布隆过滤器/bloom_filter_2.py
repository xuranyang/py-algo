import mmh3
from bitarray import bitarray

# 实现一个简单的布隆过滤器与murmurhash算法
# Bloom filter用于检查集合中是否存在某个元素，在大数据情况下具有良好的性能
# 它的 positive rate 取决于哈希函数和元素计数


BIT_SIZE = 5000000


class BloomFilter:

    def __init__(self):
        # Initialize bloom filter, set size and all bits to 0
        bit_array = bitarray(BIT_SIZE)
        bit_array.setall(0)

        self.bit_array = bit_array

    def add(self, url):
        # Add a url, and set points in bitarray to 1 (Points count is equal to hash funcs count)
        # Here use 7 hash functions.
        point_list = self.get_postions(url)

        for b in point_list:
            self.bit_array[b] = 1

    def contains(self, url):
        # Check if a url is in a collection
        point_list = self.get_postions(url)

        result = True
        for b in point_list:
            result = result and self.bit_array[b]

        return result

    def get_postions(self, url):
        # Get points positions in bit vector.
        point1 = mmh3.hash(url, 41) % BIT_SIZE
        point2 = mmh3.hash(url, 42) % BIT_SIZE
        point3 = mmh3.hash(url, 43) % BIT_SIZE
        point4 = mmh3.hash(url, 44) % BIT_SIZE
        point5 = mmh3.hash(url, 45) % BIT_SIZE
        point6 = mmh3.hash(url, 46) % BIT_SIZE
        point7 = mmh3.hash(url, 47) % BIT_SIZE

        return [point1, point2, point3, point4, point5, point6, point7]


if __name__ == '__main__':
    bloom_filter = BloomFilter()
    animals = ['dog', 'cat', 'giraffe', 'fly', 'mosquito', 'horse', 'eagle',
               'bird', 'bison', 'boar', 'butterfly', 'ant', 'anaconda', 'bear',
               'chicken', 'dolphin', 'donkey', 'crow', 'crocodile']
    # 将列表中的每个元素都添加到布隆过滤器中
    for animal in animals:
        bloom_filter.add(animal)

    # 这些元素全都在布隆过滤器中,所以布隆过滤器应该会全部判断成可能存在
    # 如果被布隆过滤器判断成一定不存在,说明一定出现了问题
    for animal in animals:
        if bloom_filter.contains(animal):
            print('[IN]{}符合预期在布隆过滤器中'.format(animal))
        else:
            print('Something is terribly went wrong for {}'.format(animal))
            print('FALSE NEGATIVE!')

    # 这些元素其实都不在布隆过滤器中，但是有几率出现误判,可能某几个元素会被误认为在布隆过滤器中
    other_animals = ['badger', 'cow', 'pig', 'sheep', 'bee', 'wolf', 'fox',
                     'whale', 'shark', 'fish', 'turkey', 'duck', 'dove',
                     'deer', 'elephant', 'frog', 'falcon', 'goat', 'gorilla',
                     'hawk']
    for other_animal in other_animals:
        if bloom_filter.contains(other_animal):
            print('[False Positive]{}其实不在布隆过滤器中,但是出现了误判'.format(other_animal))
        else:
            print('[NOT IN]{}符合预期不在布隆过滤器中'.format(other_animal))
