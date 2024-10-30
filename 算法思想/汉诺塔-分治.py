from typing import List

"""
哈诺塔问题-分治+递归
分治是一种算法思想,递归是实现这种思想的一种手段(也可以不是递归)

将规模为 i 的汉诺塔问题记作 f(i) 
f(1):即当只有一个圆盘时，我们将它直接从 A 移动至 C 即可
f(2):将两个圆盘借助 B 从 A 移至 C; 其中，C 称为目标柱、B 称为缓冲柱;
    1.先将上面的小圆盘从 A 移至 B                     -> f(1);move(A,B)
    2.再将大圆盘从 A 移至 C                         -> f(1);move(A,C)
    3.最后将小圆盘从 B 移至 C                        -> f(1);move(B,C)
f(3): dfs(3,A,B,C)
    1.令 B 为目标柱、C 为缓冲柱，将两个圆盘从 A 移至 B    -> 类似f(2); dfs(2,A,C,B)
    2.将 A 中剩余的一个圆盘从 A 直接移动至 C            -> 类似f(1); move(A,C)
    3.令 C 为目标柱、A 为缓冲柱，将两个圆盘从 B 移至 C    -> 类似f(2); dfs(2,B,A,C)
    
将问题f(3)划分为:两个子问题f(2) 和 一个子问题f(1)
 
解决汉诺塔问题的分治策略：将原问题f(n)划分为 两个子问题f(n-1) 和一个子问题f(1)
并按照以下顺序解决这三个子问题:
    1.将 n-1 个圆盘借助 C 从 A 移至 B    ->f(n-1); dfs(n-1,A,C,B)
    2.将剩余 1 个圆盘从 A 直接移至 C     ->f(1); move(A,C)
    3.将 n-1 个圆盘借助 A 从 B 移至 C    ->f(n-1); dfs(n-1,B,A,C)

对于这两个子问题f(n-1)可以通过相同的方式进行递归划分，直至达到最小子问题f(1)
步数为2^n-1
"""


def count_steps():
    """
    仅用于统计累计步数
    :return:
    """
    global steps
    steps += 1


def move(src: List[int], tar: List[int]):
    """移动一个圆盘"""
    # 从 src 顶部拿出一个圆盘
    pan = src.pop()
    # 将圆盘放入 tar 顶部
    tar.append(pan)
    # 统计累计步数
    count_steps()


def dfs(i: int, src: List[int], buf: List[int], tar: List[int]):
    """求解汉诺塔问题 f(i)"""
    # 若 src 只剩下一个圆盘，则直接将其移到 tar
    if i == 1:
        move(src, tar)
        return
    # 子问题 f(i-1) ：将 src 顶部 i-1 个圆盘借助 tar 移到 buf
    dfs(i - 1, src, tar, buf)
    # 子问题 f(1) ：将 src 剩余一个圆盘移到 tar
    move(src, tar)
    # 子问题 f(i-1) ：将 buf 顶部 i-1 个圆盘借助 src 移到 tar
    dfs(i - 1, buf, src, tar)


def solve_hanota(A: List[int], B: List[int], C: List[int]):
    """求解汉诺塔问题"""
    n = len(A)
    # 将 A 顶部 n 个圆盘借助 B 移到 C
    dfs(n, A, B, C)


if __name__ == "__main__":
    steps = 0
    # 列表尾部是柱子顶部
    # A = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    A = [5, 4, 3, 2, 1]
    B = []
    C = []
    print("初始状态下：")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")

    solve_hanota(A, B, C)

    print("圆盘移动完成后：")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")

    print(f"Total Steps: {steps}")
