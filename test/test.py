import multiprocessing as mp
import time
from multiprocessing.dummy import freeze_support


class TestClass2:
    def __del__(self):
        print("TestClass2 is deleted")


class TestClass:
    def __init__(self):
        self.x = TestClass2()

    def __del__(self):
        print("TestClass is deleted")


def test():
    print('test')
    t = TestClass()
    print('TestClass created')
    print('test end')
    return t.x


if __name__ == '__main__':
    x = test()
    print('return to main')
