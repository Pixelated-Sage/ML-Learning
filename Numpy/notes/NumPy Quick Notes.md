## 🔹 What is NumPy?
- NumPy (Numerical Python) is a Python library for numerical computing.
- Key feature: `ndarray`, a powerful N-dimensional array object.

---

## 🔹 Installation
```bash
pip install numpy
```

---

## 🔹 Importing NumPy
``` python
import numpy as np
```

---

## 🔹 Creating Arrays

### 1D, 2D, 3D Arrays
```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

### Using Functions
```python
np.zeros((2, 3))      # All zeros
np.ones((2, 3))       # All ones
np.full((2, 3), 7)    # All 7s
np.eye(3)             # Identity matrix
np.arange(0, 10, 2)   # Range with step
np.linspace(0, 1, 5)  # 5 values from 0 to 1
```

---

## 🔹 Array Attributes
```python
a.shape       # Dimensions
a.ndim        # No. of dimensions
a.size        # Total elements
a.dtype       # Data type
a.itemsize    # Size of one element in bytes
```

---

## 🔹 Reshaping & Flattening
```python
a.reshape(2, 3)
a.ravel()        # Flatten
a.flatten()
```

---

## 🔹 Indexing & Slicing
```python
a[1], a[1,2]
a[1:4], a[:, 0], a[::2]
a[1:3, 0:2]
```

### Boolean Indexing
```python
a[a > 2]
```

---

## 🔹 Array Math

### Element-wise Operations
```python
a + b, a - b, a * b, a / b
np.add(a, b)
np.multiply(a, b)
```

### Matrix Operations
```python
a.dot(b) or np.dot(a, b)
np.matmul(a, b)
```

---

## 🔹 Aggregations
```python
a.sum(), a.min(), a.max()
np.mean(a), np.std(a), np.var(a)
a.cumsum(), a.cumprod()
a.argmin(), a.argmax()
```

---

## 🔹 Axis Parameter
```python
a.sum(axis=0)   # Column-wise
a.sum(axis=1)   # Row-wise
```

---

## 🔹 Broadcasting
- Allows operations between arrays of different shapes.
```python
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
a + b  # Broadcasting happens
```

---

## 🔹 Copy vs View
```python
b = a.copy()    # New object
b = a.view()    # Shallow copy
```

---

## 🔹 Random Module
```python
np.random.rand(2, 2)
np.random.randn(3, 3)
np.random.randint(0, 10, (2, 3))
np.random.seed(42)   # For reproducibility
```

---

## 🔹 Useful Functions
```python
np.unique(a)
np.sort(a)
np.where(a > 2)
np.count_nonzero(a)
np.any(a > 3), np.all(a < 10)
np.clip(a, min, max)
```

---

## 🔹 File I/O
```python
np.save('array.npy', a)
np.load('array.npy')
np.savetxt('data.txt', a, delimiter=',')
np.genfromtxt('data.txt', delimiter=',')
```

---

## 🔹 Misc
```python
np.transpose(a)
np.vstack([a, b])
np.hstack([a, b])
```

---

## 🔹 Common Errors
- Shape mismatch in operations
- Confusion between `.copy()` and view
- Using `==` for float comparison (use `np.isclose()`)

---



