- Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with "relationsal" or "labeled" data both easy and intuitive.
- Here are just a few of the things that pandas does well:
	it has functions for analyzing, cleaning , exploring, and manipulating data.
- The name "Pandas" has a reference to both "Panel Data" and "Python data Analysis" and was created by Wes McKinnney in 2008.

### Pandas Applications
- Easy handling of missing data
- Size Mutability : columns can be inserted and deleted from DataFrames and higher dimensional objects.
- Automatic and explicit data alignment: Objects can be ecplicitly aligned to a set of labels, or the user can simply ignore the labels and let Series, DataFrame, etc.
- Automatically align the data for you in computations.
- Powerful, flexible group by functionality.
- Intelligent label-based slicing, fancy indexing, and subsetting of large data sets.
- intuitive merging and joining data sets.
- Flexible reshaping and pivoting of data sets.

### Data Structures
The best way to think about the pandas data structure is as flexible container for lower dimensional data. For example, DataFrame is a container for Series, and Series is a container for scalars. We would like to be able to insert and remove objects from these containers in a dictionary-like fashion.

**Series**:
Pandas Series is a one-Dimensional labeled array capable of holding data of any type(integer, string , float, python objects, etc). The axis labels are collectively called index. Pandas Series is nothing but a column in an excel sheet.

The object supports both integer and label-based indexing and provides a host of methods for performing operations involving the index. 

**Data Frame**:
Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and column). 
A DataFrame is a two-dimensional data structure. i.e., data is aligned in a tabular fashion in rows and columns. 
Pandas DataFrame consist of three principle components,
the data , rows , and columns.