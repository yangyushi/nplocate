# distutils: language=c++
"""
A collection of auxiliary functions written in cython for efficiency
"""
import cython
from cython.operator cimport preincrement as inc
from libcpp cimport bool
from libcpp.set cimport set
from libcpp.vector cimport vector


cdef bool should_join(vector[int]& p1, vector[int]& p2):
    for i in p1:
        for j in p2:
            if i == j:
                return True
    return False


cdef vector[int] join_pair(vector[int]& p1, vector[int]& p2):
    cdef set[int] unique
    cdef vector[int] joined
    for i in p1:
        unique.insert(i)
    for j in p2:
        unique.insert(j)
    for u in unique:
        joined.push_back(u)
    return joined


cdef void join_pairs_inplace(vector[vector[int]]& pairs):
    """
    Example:
        >>> pairs = [(2, 3), (3, 5), (2, 6), (8, 9), (9, 10)]
        >>> join_pairs(pairs)
        [(2, 3, 5, 6), (8, 9, 10)]
    """
    cdef int length = pairs.size()
    cdef vector[int] p1, p2
    cdef vector[vector[int]].iterator it
    cdef int i1, i2
    for i1 in range(length):
        for i2 in range(length):
            p1 = pairs[i1]
            p2 = pairs[i2]
            if (i1 != i2) and should_join(p1, p2):
                pairs.push_back(join_pair(p1, p2))

                # erase i1
                it = pairs.begin()
                for _ in range(i1):
                    inc(it)
                pairs.erase(it)

                # erase i2
                it = pairs.begin()
                for _ in range(i2 - 1):
                    inc(it)
                pairs.erase(it)

                join_pairs_inplace(pairs)
                return
    return


def join_pairs(const vector[vector[int]] pairs):
    cdef vector[vector[int]] joined = pairs;
    join_pairs_inplace(joined)
    return joined
