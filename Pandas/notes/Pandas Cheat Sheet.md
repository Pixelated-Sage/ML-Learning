# 📊 Pandas Cheat Sheet

Pandas is a powerful Python library for data manipulation and analysis.

---

## 🔹 Installation
```bash
pip install pandas
```

---

## 🔹 Importing
```python
import pandas as pd
```

---

## 🔹 Creating Data Structures

### Series
```python
s = pd.Series([1, 3, 5, np.nan, 6])
```

### DataFrame
```python
df = pd.DataFrame({
    "A": 1.,
    "B": pd.Timestamp("20230101"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["test", "train", "test", "train"]),
    "F": "foo"
})
```

---

## 🔹 Reading / Writing Data

### CSV
```python
pd.read_csv("file.csv")
df.to_csv("file.csv")
```

### Excel
```python
pd.read_excel("file.xlsx")
df.to_excel("file.xlsx")
```

---

## 🔹 Data Inspection
```python
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.columns
df.index
df.dtypes
```

---

## 🔹 Selecting Data

### By Column
```python
df["A"]
df.A
```

### By Row
```python
df.loc[0]      # Label-based
df.iloc[0]     # Index-based
```

### Subset
```python
df.loc[0, "A"]
df.iloc[0, 0]
df[["A", "B"]]
df[0:3]
```

---

## 🔹 Filtering / Boolean Indexing
```python
df[df["A"] > 0]
df[(df["A"] > 0) & (df["B"] < 5)]
```

---

## 🔹 Setting Data
```python
df.at[0, "A"] = 100
df.iat[0, 0] = 200
df["A"] = df["A"].astype(float)
```

---

## 🔹 Missing Data
```python
df.dropna()
df.fillna(0)
df.isna()
df.notna()
```

---

## 🔹 Operations

### Stats & Aggregations
```python
df.mean()
df.sum()
df.count()
df.min()
df.max()
```

### Apply
```python
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())
```

---

## 🔹 Sorting
```python
df.sort_index(axis=1, ascending=False)
df.sort_values(by="B")
```

---

## 🔹 Grouping
```python
df.groupby("E").sum()
df.groupby(["E", "F"]).mean()
```

---

## 🔹 Merging / Joining

### Concat
```python
pd.concat([df1, df2])
```

### Merge
```python
pd.merge(df1, df2, on="key")
```

### Join
```python
df1.join(df2, lsuffix="_left", rsuffix="_right")
```

---

## 🔹 Time Series
```python
pd.date_range("20230101", periods=6)
df.index = pd.to_datetime(df.index)
df.resample("M").mean()
```

---

## 🔹 Exporting
```python
df.to_csv("data.csv")
df.to_excel("data.xlsx")
df.to_json("data.json")
```

---

## 🔹 Common Errors
- KeyError for wrong column names
- Chained indexing issues (use `.loc`)
- Shape mismatch in assignments
