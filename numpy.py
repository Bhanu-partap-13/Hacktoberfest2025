import numpy as np
import os, tempfile

print('===== 1. Array creation =====')
a = np.array([1, 2, 3])                    # from list
b = np.arange(0, 10, 2)                    # start, stop, step
c = np.linspace(0, 1, 5)                   # 5 nums 0→1
d = np.ones((3, 4))                        # 3×4 ones
e = np.zeros((2, 3))                       # zeros
f = np.eye(3)                              # identity matrix
g = np.diag([1, 2, 3])                     # diagonal matrix
h = np.random.rand(2, 3)                   # uniform [0,1)
i = np.random.randn(2, 3)                  # standard normal
j = np.empty((2, 2))                       # uninitialised
k = np.full((2, 2), 7)                     # constant
l = np.identity(3)                         # same as eye
print('shapes:', a.shape, d.shape, f.shape)

print('\n===== 2. Attributes =====')
arr = np.random.rand(3, 4)
print('ndim:', arr.ndim, 'shape:', arr.shape, 'size:', arr.size,
      'dtype:', arr.dtype, 'itemsize:', arr.itemsize, 'nbytes:', arr.nbytes)

print('\n===== 3. Reshaping & axes =====')
m = np.arange(12)
print('original:', m)
print('reshape (3,4):', m.reshape(3, 4))
print('ravel:', m.reshape(3, 4).ravel())          # flatten
print('transpose:', m.reshape(3, 4).T)
print('flatten (copy):', m.reshape(3, 4).flatten())
print('squeeze:', np.array([[[1, 2, 3]]]).squeeze())  # removes size-1 dims
print('expand_dims:', np.expand_dims(m, axis=0).shape)
print('newaxis:', m[np.newaxis, :].shape, m[:, np.newaxis].shape)

print('\n===== 4. Indexing & slicing =====')
n = np.arange(10)**2
print('array:', n)
print('n[2:7:2] =', n[2:7:2])               # start:stop:step
print('n[::-1]  =', n[::-1])                 # reverse
o = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('2-D slice o[1:, :2] =\n', o[1:, :2])
print('fancy indexing o[[0,2], [1,2]] =', o[[0, 2], [1, 2]])  # (0,1)&(2,2)
print('boolean mask n[n>20] =', n[n > 20])

print('\n===== 5. Broadcasting =====')
a = np.array([[0, 1, 2], [3, 4, 5]])    # (2,3)
b = np.array([10, 20, 30])              # (3,)
print('a + b (broadcast) =\n', a + b)

print('\n===== 6. Universal functions (ufuncs) =====')
x = np.arange(1, 6)
print('x        :', x)
print('x + 10   :', x + 10)
print('np.sqrt  :', np.sqrt(x))
print('np.exp   :', np.exp(x))
print('np.sin   :', np.sin(x))
print('np.log   :', np.log(x))

print('\n===== 7. Reductions =====')
print('sum, mean, std, var:', x.sum(), x.mean(), x.std(), x.var())
print('min/max:', x.min(), x.max())
print('argmin/argmax idx:', x.argmin(), x.argmax())
print('cumsum:', x.cumsum())
print('allclose:', np.allclose([1e10, 1e-7], [1.00001e10, 1e-8]))

print('\n===== 8. Axis-wise ops =====')
mat = np.arange(1, 13).reshape(3, 4)
print('matrix:\n', mat)
print('col sum (axis=0):', mat.sum(axis=0))
print('row sum (axis=1):', mat.sum(axis=1))

print('\n===== 9. Sorting & unique =====')
arr = np.array([3, 1, 2, 3])
print('sort:', np.sort(arr))
print('argsort idx:', np.argsort(arr))
print('unique values:', np.unique(arr))
print('unique + inverse + counts:', np.unique(arr, return_inverse=True, return_counts=True))

print('\n===== 10. Copy vs view =====')
v = np.arange(5)
w = v          # reference
x = v.copy()   # independent copy
w[0] = 99
print('v (changed):', v, 'x (copy untouched):', x)

print('\n===== 11. Where (vectorised if) =====')
a = np.array([1, 4, 2, 8])
b = np.where(a > 3, a, -1)          # condition, x_if_true, x_if_false
print('where result:', b)

print('\n===== 12. Concatenate & stack =====')
p = np.array([[1, 2], [3, 4]])
q = np.array([[5, 6]])
print('vstack:\n', np.vstack([p, q]))
print('hstack:\n', np.hstack([p, q.T]))
print('column_stack:\n', np.column_stack([p, q.T]))

print('\n===== 13. Splitting an array =====')
big = np.arange(10)
print('array_split (3 parts):', np.array_split(big, 3))
print('split (equal size):', np.array_split(big, 2))

print('\n===== 14. Saving / loading =====')
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    np.save(tmp.name, big)
    loaded = np.load(tmp.name + '.npy')
    print('round-trip save/load:', np.array_equal(big, loaded))
    os.remove(tmp.name + '.npy')

print('\n===== 15. Datetime & linear algebra extras =====')
# datetime
dates = np.arange('2021-01', '2021-02', dtype='datetime64[D]')
print('datetime array len:', len(dates))

# dot product
v1 = np.array([1, 2])
v2 = np.array([3, 4])
print('dot v1·v2:', np.dot(v1, v2))

# matrix multiply
A = np.arange(4).reshape(2, 2)
B = np.ones((2, 2))
print('A @ B:\n', A @ B)
print('det(A):', np.linalg.det(A.astype(float)))
print('inv(A):', np.linalg.inv(A.astype(float)))
print('eigvals(A):', np.linalg.eigvals(A.astype(float)))

print('\n===== 16. Random sampling & seeding =====')
np.random.seed(42)
print('randint(0,10,5):', np.random.randint(0, 10, 5))
print('choice w/ prob:', np.choice(['A', 'B'], 5, p=[0.9, 0.1]))
print('shuffle in-place:')
arr = np.arange(5)
np.random.shuffle(arr)
print(arr)

print('\n===== 17. Performance tip (vectorisation) =====')
size = 1_000_000
a = np.random.rand(size)
b = np.random.rand(size)

# vectorised (fast)
c = a + b

# python loop (slow) – commented out
# d = np.array([x + y for x, y in zip(a, b)])

print('Vectorised shape:', c.shape)

print('\n===== 18. One-liner cheat reminder =====')
reminder = """
array, arange, linspace, ones, zeros, eye, diag
reshape, ravel, T, squeeze, expand_dims
shape, ndim, size, dtype, nbytes
Indexing: [start:stop:step], fancy, boolean
Broadcasting: automatic size alignment
ufuncs: np.sqrt, np.exp, np.sin, np.log, np.sum, np.mean, np.std
axis: 0=column-wise, 1=row-wise
sort, argsort, unique, where
copy vs view: use .copy() to avoid side-effects
vstack, hstack, column_stack, split, array_split
save/load: np.save/np.load, np.savetxt/np.loadtxt
dot / @, det, inv, eigvals
random: rand, randn, randint, choice, shuffle
vectorisation > python loops
"""
print(reminder)
