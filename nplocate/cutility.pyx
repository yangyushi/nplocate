# distutils: language=c++
"""
A collection of auxiliary functions written in cython for efficiency
"""
import cython
from cython.operator cimport preincrement as inc
from libcpp cimport bool
from libcpp.set cimport set
from libcpp.vector cimport vector


"""cpp equivalent
#include <vector>
#include <set>
using namespace std;

using Pair = vector<int>;
using Pairs = vector<Pair>;

bool should_join(Pair p1, Pair p2){
    for (int n1 : p1){
    for (int n2 : p2){
        if (n1 == n2) {return true;}
    }}
    return false;
}

Pair join_pair(Pair p1, Pair p2){
    set<int> unique;
    Pair joined;
    for (int num : p1) {unique.insert(num);}
    for (int num : p2) {unique.insert(num);}
    copy(unique.begin(), unique.end(), back_inserter(joined));
    return joined;
}


void join_pairs_inplace(Pairs& pairs){
    int length = pairs.size();
    Pair p1, p2;
    for (int i1 = 0; i1 < length; i1++){
        for (int i2 = 0; i2 < length; i2++){
            p1 = pairs[i1];
            p2 = pairs[i2];
            if ((i1 != i2) and (should_join(p1, p2))){
                pairs.push_back(join_pair(p1, p2));
                pairs.erase(pairs.begin() + i1);
                pairs.erase(pairs.begin() + i2 - 1);
                join_pairs_inplace(pairs);
                return;
            }
        }
    }
}

Pairs join_pairs(Pairs& pairs){
    Pairs joined;
    copy(pairs.begin(), pairs.end(), back_inserter(joined));
    join_pairs_inplace(joined);
    return joined;
}

"""

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
