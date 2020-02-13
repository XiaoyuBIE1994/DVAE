#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is used to test some python grammar
"""

class A (object):
    def __init__(self, a):
        self.a = a

    def fun(self):
        print('1')

class B (A):
    def __init__(self, a, b):
        super(B, self).__init__(a)
        self.b = b
        
    def fun(self):
        print('2')

if __name__ == '__main__':
    b = B(3, 4)
    b.fun()
    print(b.a)
    print(b.b)