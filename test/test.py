import sys
sys.path.insert(0, '../nplocate')
import numpy as np
from cutility import join_pairs
from utility import join_pairs_py
from time import time

N = 500
pairs = []
indices = np.random.randint(0, N, size=2 * N)
for i in range(N):
    pairs.append((indices[i], indices[i+1]))

t0 = time()
joined_py = join_pairs_py(pairs)
t_py = time() - t0

t0 = time()
joined = join_pairs(pairs)
t_cpp = time() - t0


for p1, p2 in zip(joined_py, joined):
    assert np.all(np.array(p1) == np.array(p2)), "inconsistent result"

print(f"CPP: {t_cpp} s", )
print(f"PY : {t_py} s")
