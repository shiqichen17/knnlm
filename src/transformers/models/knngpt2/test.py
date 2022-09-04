import numpy as np
keys = np.memmap('a_keys.npy', dtype=np.float16, mode='r+', shape=(119368369, 768))
vals = np.memmap('a_vals.npy', dtype=np.int, mode='r+', shape=(119368369, 1))
#print(keys[-2000:-1000])
print(vals[:10])
print(vals[511])
print(vals[511:521])