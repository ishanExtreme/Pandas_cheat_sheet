<h1 align='center'><u>Basic Pandas</u></h1>

## Why Use Pandas?
### The recent success of machine learning algorithms is partly due to the huge amounts of data that we have available to train our algorithms on. However, when it comes to data, quantity is not the only thing that matters, the quality of your data is just as important. It often happens that large datasets don’t come ready to be fed into your learning algorithms. More often than not, large datasets will often have missing values, outliers, incorrect values, etc… Having data with a lot of missing or bad values, for example, is not going to allow your machine learning algorithms to perform well. Therefore, one very important step in machine learning is to look at your data first and make sure it is well suited for your training algorithm by doing some basic data analysis. This is where Pandas come in. Pandas Series and DataFrames are designed for fast data analysis and manipulation, as well as being flexible and easy to use. Below are just a few features that makes Pandas an excellent package for data analysis:

1. Allows the use of labels for rows and columns
2. Can calculate rolling statistics on time series data
3. Easy handling of NaN values
4. Is able to load data of different formats into DataFrames
5. Can join and merge different datasets together
6. It integrates with NumPy and Matplotlib

### For these and other reasons, Pandas DataFrames have become one of the most commonly used Pandas object for data analysis in Python.

<h1 style='color:red'>Note:pandas inherit from numpy and hence features like numpy boolean indexing and other operations of numpy arrays can be applied here too</h1>

<h2 align='center'>A.Pandas series</h2>

### pandas.Series(data,index)
<h3 style='color:green'>important poitnts<br>1.<u><b>one-dimensional</b></u> <u>array-like</u> object<br>2.can hold data of different data types(but in numpy arrays all data types are converted into one)</h3>


```python
import pandas as pd

groceries = pd.Series(data = [30, 6, 'Yes', 'No'], index = ['eggs', 'apples', 'milk', 'bread'])

groceries
```




    eggs       30
    apples      6
    milk      Yes
    bread      No
    dtype: object




```python
# We print some information about Groceries
print('Groceries has shape:', groceries.shape)
print('Groceries has dimension:', groceries.ndim)
print('Groceries has a total of', groceries.size, 'elements')
print()
# We print the index and data of Groceries
print('The data in Groceries is:', groceries.values)
print('The index of Groceries is:', groceries.index)
```

    Groceries has shape: (4,)
    Groceries has dimension: 1
    Groceries has a total of 4 elements
    
    The data in Groceries is: [30 6 'Yes' 'No']
    The index of Groceries is: Index(['eggs', 'apples', 'milk', 'bread'], dtype='object')



```python
# We check whether bread is a food item (an index) in Groceries
print('bread' in groceries)#check index not data
```

    True


### Manipulating pandas series


```python
#Accessing elements using labes
print(groceries['eggs'])
print()
print(groceries[['eggs','milk']])
print()
print(groceries.loc[['eggs','apples']])
print()
#Accessing elemts using integer index
print(groceries[0])
print()
print(groceries[[0,2]])
print()
print(groceries.iloc[[0,1]])
```

    30
    
    eggs     30
    milk    Yes
    dtype: object
    
    eggs      30
    apples     6
    dtype: object
    
    30
    
    eggs     30
    milk    Yes
    dtype: object
    
    eggs      30
    apples     6
    dtype: object


### to remove any ambiguity to whether we are referring to an index label or numerical index use loc and iloc

### pandas series are mutable and hence can be modified


```python
# We display the original grocery list
print('Original Grocery List:\n', groceries)

# We change the number of eggs to 2
groceries['eggs'] = 2

# We display the changed grocery list
print()
print('Modified Grocery List:\n', groceries)
```

    Original Grocery List:
     eggs       30
    apples      6
    milk      Yes
    bread      No
    dtype: object
    
    Modified Grocery List:
     eggs        2
    apples      6
    milk      Yes
    bread      No
    dtype: object


### Series.drop(label, inplace=False)


```python
# We display the original grocery list
print('Original Grocery List:\n', groceries)

# We remove apples from our grocery list. The drop function removes elements out of place
print()
print('We remove apples (out of place):\n', groceries.drop('apples'))

# When we remove elements out of place the original Series remains intact. To see this
# we display our grocery list again
print()
print('Grocery List after removing apples out of place is still intact:\n', groceries)
```

    Original Grocery List:
     eggs        2
    apples      6
    milk      Yes
    bread      No
    dtype: object
    
    We remove apples (out of place):
     eggs       2
    milk     Yes
    bread     No
    dtype: object
    
    Grocery List after removing apples out of place is still intact:
     eggs        2
    apples      6
    milk      Yes
    bread      No
    dtype: object


### if we set inplace to True than original list is also modified


```python
groceries.drop('apples', inplace = True)
print(groceries)
```

    eggs       2
    milk     Yes
    bread     No
    dtype: object


### Arithmetic operations


```python
fruits= pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])
fruits
```




    apples     10
    oranges     6
    bananas     3
    dtype: int64




```python
# We print fruits for reference
print('Original grocery list of fruits:\n ', fruits)

# We perform basic element-wise operations using arithmetic symbols
print()
print('fruits + 2:\n', fruits + 2) # We add 2 to each item in fruits
print()
print('fruits - 2:\n', fruits - 2) # We subtract 2 to each item in fruits
print()
print('fruits * 2:\n', fruits * 2) # We multiply each item in fruits by 2 
print()
print('fruits / 2:\n', fruits / 2) # We divide each item in fruits by 2
print()
```

    Original grocery list of fruits:
      apples     10
    oranges     6
    bananas     3
    dtype: int64
    
    fruits + 2:
     apples     12
    oranges     8
    bananas     5
    dtype: int64
    
    fruits - 2:
     apples     8
    oranges    4
    bananas    1
    dtype: int64
    
    fruits * 2:
     apples     20
    oranges    12
    bananas     6
    dtype: int64
    
    fruits / 2:
     apples     5.0
    oranges    3.0
    bananas    1.5
    dtype: float64
    


#### Applying mathemetical functions from numpy


```python
import numpy as np

# We print fruits for reference
print('Original grocery list of fruits:\n', fruits)

# We apply different mathematical functions to all elements of fruits
print()
print('EXP(X) = \n', np.exp(fruits))
print() 
print('SQRT(X) =\n', np.sqrt(fruits))
print()
print('POW(X,2) =\n',np.power(fruits,2))
```

    Original grocery list of fruits:
     apples     10
    oranges     6
    bananas     3
    dtype: int64
    
    EXP(X) = 
     apples     22026.465795
    oranges      403.428793
    bananas       20.085537
    dtype: float64
    
    SQRT(X) =
     apples     3.162278
    oranges    2.449490
    bananas    1.732051
    dtype: float64
    
    POW(X,2) =
     apples     100
    oranges     36
    bananas      9
    dtype: int64


#### On single elements


```python
# We print fruits for reference
print('Original grocery list of fruits:\n ', fruits)
print()

# We add 2 only to the bananas
print('Amount of bananas + 2 = ', fruits['bananas'] + 2)
print()

# We subtract 2 from apples
print('Amount of apples - 2 = ', fruits.iloc[0] - 2)
print()

# We multiply apples and oranges by 2
print('We double the amount of apples and oranges:\n', fruits[['apples', 'oranges']] * 2)
print()

# We divide apples and oranges by 2
print('We half the amount of apples and oranges:\n', fruits.loc[['apples', 'oranges']] / 2)
```

    Original grocery list of fruits:
      apples     10
    oranges     6
    bananas     3
    dtype: int64
    
    Amount of bananas + 2 =  5
    
    Amount of apples - 2 =  8
    
    We double the amount of apples and oranges:
     apples     20
    oranges    12
    dtype: int64
    
    We half the amount of apples and oranges:
     apples     5.0
    oranges    3.0
    dtype: float64


#### operation on series with mixed data types
#### be carefull to check whether the operator applies to all selected elements


```python
print(groceries)
print()
print(groceries*2)#multiplication on string doubles it 
#if we apply '/' than there will be error as string dont have '/' operator
```

    eggs       2
    milk     Yes
    bread     No
    dtype: object
    
    eggs          4
    milk     YesYes
    bread      NoNo
    dtype: object


<h1 align='center'>B.Pandas DataFrames</h1>

<h3 style='color:green'>1.think of Pandas DataFrames as being similar to a spreadsheet.<br>2.two-dimensional data structures with labeled rows and columns.<br>3.can hold many data types.


#### Creating By Dictionary

### using pandas series in dictionary


```python
# We import Pandas as pd into Python
import pandas as pd

# We create a dictionary of Pandas Series 
items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

# We print the type of items to see that it is a dictionary
print(type(items))
```

    <class 'dict'>



```python
# We create a Pandas DataFrame by passing it a dictionary of Pandas Series
shopping_carts = pd.DataFrame(items)

# We display the DataFrame rows are alphabetically arranged
shopping_carts

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bob</th>
      <th>Alice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bike</th>
      <td>245.0</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>book</th>
      <td>NaN</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>glasses</th>
      <td>NaN</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>pants</th>
      <td>25.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>watch</th>
      <td>55.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### NaN stands for Not a Number, and is Pandas way of indicating that it doesn't have a value for that particular row and column index.
### If we don't provide index labels to the Pandas Series, Pandas will use numerical row indexes when it creates the DataFrame.


```python
# We create a dictionary of Pandas Series without indexes
data = {'Bob' : pd.Series([245, 25, 55]),
        'Alice' : pd.Series([40, 110, 500, 45])}

# We create a DataFrame
df = pd.DataFrame(data)

# We display the DataFrame
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bob</th>
      <th>Alice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>245.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.0</td>
      <td>110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We print some information about shopping_carts
print('shopping_carts has shape:', shopping_carts.shape)
print('shopping_carts has dimension:', shopping_carts.ndim)
print('shopping_carts has a total of:', shopping_carts.size, 'elements')
print()
print('The data in shopping_carts is:\n', shopping_carts.values)
print()
print('The row index in shopping_carts is:', shopping_carts.index)
print()
print('The column index in shopping_carts is:', shopping_carts.columns)
```

    shopping_carts has shape: (5, 2)
    shopping_carts has dimension: 2
    shopping_carts has a total of: 10 elements
    
    The data in shopping_carts is:
     [[245. 500.]
     [ nan  40.]
     [ nan 110.]
     [ 25.  45.]
     [ 55.  nan]]
    
    The row index in shopping_carts is: Index(['bike', 'book', 'glasses', 'pants', 'watch'], dtype='object')
    
    The column index in shopping_carts is: Index(['Bob', 'Alice'], dtype='object')



```python
# We Create a DataFrame that only has Bob's data
bob_shopping_cart = pd.DataFrame(items, columns=['Bob'])

# We display bob_shopping_cart
bob_shopping_cart


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bike</th>
      <td>245</td>
    </tr>
    <tr>
      <th>pants</th>
      <td>25</td>
    </tr>
    <tr>
      <th>watch</th>
      <td>55</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We Create a DataFrame that only has selected items for both Alice and Bob
sel_shopping_cart = pd.DataFrame(items, index = ['pants', 'book'])

# We display sel_shopping_cart
sel_shopping_cart


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bob</th>
      <th>Alice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pants</th>
      <td>25.0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>book</th>
      <td>NaN</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We Create a DataFrame that only has selected items for Alice
alice_sel_shopping_cart = pd.DataFrame(items, index = ['glasses', 'bike'], columns = ['Alice'])

# We display alice_sel_shopping_cart
alice_sel_shopping_cart
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>glasses</th>
      <td>110</td>
    </tr>
    <tr>
      <th>bike</th>
      <td>500</td>
    </tr>
  </tbody>
</table>
</div>



### You can also manually create DataFrames from a dictionary of lists (arrays). The procedure is the same as before, we start by creating the dictionary and then passing the dictionary to the pd.DataFrame() function. In this case, however, all the lists (arrays) in the dictionary must be of the same length. Let' see an example:


```python
# We create a dictionary of lists (arrays)
data = {'Integers' : [1,2,3],
        'Floats' : [4.5, 8.2, 9.6]}

# We create a DataFrame 
df = pd.DataFrame(data)

# We display the DataFrame
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Integers</th>
      <th>Floats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We create a dictionary of lists (arrays)
data = {'Integers' : [1,2,3],
        'Floats' : [4.5, 8.2, 9.6]}

# We create a DataFrame and provide the row index
df = pd.DataFrame(data, index = ['label 1', 'label 2', 'label 3'])

# We display the DataFrame
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Integers</th>
      <th>Floats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>label 1</th>
      <td>1</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>label 2</th>
      <td>2</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>label 3</th>
      <td>3</td>
      <td>9.6</td>
    </tr>
  </tbody>
</table>
</div>



### using list of dictionery


```python
# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame 
store_items = pd.DataFrame(items2)

# We display the DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])

# We display the DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>



### Manipulating elements

### dataframe[column][row]


```python
# We print the store_items DataFrame
print(store_items)

# We access rows, columns and elements using labels
print()
print('How many bikes are in each store:\n', store_items[['bikes']])
print()
print('How many bikes and pants are in each store:\n', store_items[['bikes', 'pants']])
print()
print('What items are in Store 1:\n', store_items.loc[['store 1']])
print()
print('How many bikes are in Store 2:', store_items['bikes']['store 2'])#[columns][rows]

```

             bikes  pants  watches  glasses
    store 1     20     30       35      NaN
    store 2     15      5       10     50.0
    
    How many bikes are in each store:
              bikes
    store 1     20
    store 2     15
    
    How many bikes and pants are in each store:
              bikes  pants
    store 1     20     30
    store 2     15      5
    
    What items are in Store 1:
              bikes  pants  watches  glasses
    store 1     20     30       35      NaN
    
    How many bikes are in Store 2: 15



```python
# We add a new column named shirts to our store_items DataFrame indicating the number of
# shirts in stock at each store. We will put 15 shirts in store 1 and 2 shirts in store 2
store_items['shirts'] = [15,2]

# We display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
      <th>shirts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>15</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We make a new column called suits by adding the number of shirts and pants
store_items['suits'] = store_items['pants'] + store_items['shirts']

# We display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>15</td>
      <td>45</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



### Adding new row
#### append()


```python
# We create a dictionary from a list of Python dictionaries that will number of items at the new store
new_items = [{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]

# We create new DataFrame with the new_items and provide and index labeled store 3
new_store = pd.DataFrame(new_items, index = ['store 3'])

# We display the items at the new store
print(new_store)

# We append store 3 to our store_items DataFrame
store_items = store_items.append(new_store)

# We display the modified DataFrame
store_items
```

             bikes  pants  watches  glasses
    store 3     20     30       35        4





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Adding new column to specific rows


```python
# We add a new column using data from particular rows in the watches column
# this adds watches to store  and store3
store_items['new watches'] = store_items['watches'][1:]

# We display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
      <th>new watches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>



### dataframe.insert(loc,label,data)
### Adds a new column


```python
# We insert a new column with label shoes right before the column with numerical index 4
store_items.insert(4, 'shoes', [8,5,0])

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
      <th>shoes</th>
      <th>shirts</th>
      <th>suits</th>
      <th>new watches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>8</td>
      <td>15.0</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
      <td>5</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>4.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>



### Deleting elemets
### .pop() used for deleting columns only


```python
# We remove the new watches column
store_items.pop('new watches')

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>glasses</th>
      <th>shoes</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>8</td>
      <td>15.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>50.0</td>
      <td>5</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>4.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### .drop() can delete both rows or columns using axis keyword
### axis = 1 ->y-axis
### axis = 0 ->x-axis


```python
# We remove the watches and shoes columns
store_items = store_items.drop(['watches', 'shoes'], axis = 1)

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>50.0</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We remove the store 2 and store 1 rows
store_items = store_items.drop(['store 2', 'store 1'], axis = 0)

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Renaming labels
### .rename()


```python
# We change the column label bikes to hats
store_items = store_items.rename(columns = {'bikes': 'hats'})

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hats</th>
      <th>pants</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We change the row label from store 3 to last store
store_items = store_items.rename(index = {'store 3': 'last store'})

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hats</th>
      <th>pants</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>last store</th>
      <td>20</td>
      <td>30</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We change the row index to be the data in the pants column
store_items = store_items.set_index('pants')

# we display the modified DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hats</th>
      <th>glasses</th>
      <th>shirts</th>
      <th>suits</th>
    </tr>
    <tr>
      <th>pants</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>20</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Dealing with NaN values


```python
# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},
{'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},
{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

# We display the DataFrame
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>15.0</td>
      <td>8</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>2.0</td>
      <td>5</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



### Counting NaN values


```python
# We count the number of NaN values in store_items
# break it to see how it works
x =  store_items.isnull().sum().sum()

# We print x
print('Number of NaN values in our DataFrame:', x)
```

    Number of NaN values in our DataFrame: 3


#### Counting non NaN values


```python
store_items.count()
```




    bikes      3
    pants      3
    watches    3
    shirts     2
    shoes      3
    suits      2
    glasses    2
    dtype: int64



### deleting NaN value
### .dropna(axis)
### The .dropna(axis) method eliminates any rows with NaN values when axis = 0 is used and will eliminate any columns with NaN values when axis = 1 is used. Let's see some examples


```python
print(store_items)
print('-'*65)
print()
print(store_items.dropna(axis=0))
print('-'*65)
print()
print(store_items)
print('-'*65)
print()
print(store_items.dropna(axis=1))
```

             bikes  pants  watches  shirts  shoes  suits  glasses
    store 1     20     30       35    15.0      8   45.0      NaN
    store 2     15      5       10     2.0      5    7.0     50.0
    store 3     20     30       35     NaN     10    NaN      4.0
    -----------------------------------------------------------------
    
             bikes  pants  watches  shirts  shoes  suits  glasses
    store 2     15      5       10     2.0      5    7.0     50.0
    -----------------------------------------------------------------
    
             bikes  pants  watches  shirts  shoes  suits  glasses
    store 1     20     30       35    15.0      8   45.0      NaN
    store 2     15      5       10     2.0      5    7.0     50.0
    store 3     20     30       35     NaN     10    NaN      4.0
    -----------------------------------------------------------------
    
             bikes  pants  watches  shoes
    store 1     20     30       35      8
    store 2     15      5       10      5
    store 3     20     30       35     10


### Notice that the .dropna() method eliminates (drops) the rows or columns with NaN values out of place. This means that the original DataFrame is not modified. You can always remove the desired rows or columns in place by setting the keyword inplace = True inside the dropna() function.

### Filling NaN values
#### .fillna()


```python
# We replace all NaN values with 0
store_items.fillna(0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>15.0</td>
      <td>8</td>
      <td>45.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>2.0</td>
      <td>5</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>0.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



### .fillna(method = 'ffill', axis)
### ffill method to replace NaN values using the previous known value along the given axis


```python
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>15.0</td>
      <td>8</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>2.0</td>
      <td>5</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
store_items.fillna(method = 'ffill', axis = 0)#replaces with prevoius rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>15.0</td>
      <td>8</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>2.0</td>
      <td>5</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>2.0</td>
      <td>10</td>
      <td>7.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
store_items.fillna(method = 'ffill', axis = 1)#replaces with previous columns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20.0</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>45.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20.0</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>35.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



### .fillna(method = 'backfill', axis)
### backfill method to replace NaN values using the next known value along the given axis


```python
store_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>15.0</td>
      <td>8</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>2.0</td>
      <td>5</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We replace NaN values with the next value in the column
store_items.fillna(method = 'backfill', axis = 0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>15.0</td>
      <td>8</td>
      <td>45.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15</td>
      <td>5</td>
      <td>10</td>
      <td>2.0</td>
      <td>5</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20</td>
      <td>30</td>
      <td>35</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We replace NaN values with the next value in the row
store_items.fillna(method = 'backfill', axis = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bikes</th>
      <th>pants</th>
      <th>watches</th>
      <th>shirts</th>
      <th>shoes</th>
      <th>suits</th>
      <th>glasses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>store 1</th>
      <td>20.0</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>45.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>store 2</th>
      <td>15.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>store 3</th>
      <td>20.0</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



### Note we can always replace the NaN values in place by setting the keyword inplace = True inside the fillna() function.

### isnull().any()
### gives True or False if there are any NaN values in any rows


```python

```
