# **Project 1**: Mercedes-Benz Greener Manufacturing 


```python
# Import the required libraries
import numpy as np
import pandas as pd
# for dimensionality reduction
from sklearn.decomposition import PCA
```


```python
# Read the data from train.csv

df_train = pd.read_csv('train.csv')
# let us understand the data
print('Size of training set: {} rows and {} columns'.format(*df_train.shape))
df_train.head()
```

    Size of training set: 4209 rows and 378 columns





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
      <th>ID</th>
      <th>y</th>
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X8</th>
      <th>...</th>
      <th>X375</th>
      <th>X376</th>
      <th>X377</th>
      <th>X378</th>
      <th>X379</th>
      <th>X380</th>
      <th>X382</th>
      <th>X383</th>
      <th>X384</th>
      <th>X385</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>130.81</td>
      <td>k</td>
      <td>v</td>
      <td>at</td>
      <td>a</td>
      <td>d</td>
      <td>u</td>
      <td>j</td>
      <td>o</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>88.53</td>
      <td>k</td>
      <td>t</td>
      <td>av</td>
      <td>e</td>
      <td>d</td>
      <td>y</td>
      <td>l</td>
      <td>o</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>76.26</td>
      <td>az</td>
      <td>w</td>
      <td>n</td>
      <td>c</td>
      <td>d</td>
      <td>x</td>
      <td>j</td>
      <td>x</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>80.62</td>
      <td>az</td>
      <td>t</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>x</td>
      <td>l</td>
      <td>e</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>78.02</td>
      <td>az</td>
      <td>v</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>h</td>
      <td>d</td>
      <td>n</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 378 columns</p>
</div>




```python
# Collect the Y values into an array
# seperate the y from the data as we will use this to learn as the prediction output
y_train = df_train['y'].values
```


```python
# Understand the data types we have
# iterate through all the columns which has X in the name of the column
cols = [c for c in df_train.columns if 'X' in c]
print('Number of features: {}'.format(len(cols)))
print('Feature types:')
df_train[cols].dtypes.value_counts()
```

    Number of features: 376
    Feature types:





    int64     368
    object      8
    dtype: int64




```python
#Count the data in each of the columns
counts = [[], [], []]
for c in cols:
    typ = df_train[c].dtype
    uniq = len(np.unique(df_train[c]))
    if uniq == 1:
        counts[0].append(c)
    elif uniq == 2 and typ == np.int64:
        counts[1].append(c)
    else:
        counts[2].append(c)

print('Constant features: {} Binary features: {} Categorical features: {}\n'
      .format(*[len(c) for c in counts]))
print('Constant features:', counts[0])
print('Categorical features:', counts[2])
```

    Constant features: 12 Binary features: 356 Categorical features: 8
    
    Constant features: ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']
    Categorical features: ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']



```python
# Read the test.csv data
df_test = pd.read_csv('test.csv')
# remove columns ID and Y from the data as they are not used for learning
usable_columns = list(set(df_train.columns) - set(['ID', 'y']))
y_train = df_train['y'].values
id_test = df_test['ID'].values
x_train = df_train[usable_columns]
x_test = df_test[usable_columns]
```


```python
# Step 1: Check for null and unique values for test and train sets
x_train.isnull().any().any()
```




    False




```python
x_test.isnull().any().any()
```




    False



- ##### There are no missing values in the dataframes


```python
# Step 2: If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
# Step 3: Apply label encoder

for column in usable_columns:
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1) # Column with only one 
        # value is useless so we drop it
        x_test.drop(column, axis=1)
    if cardinality > 2: # Column is categorical
        mapper = lambda x: sum([ord(digit) for digit in x])
        x_train[column] = x_train[column].apply(mapper)
        x_test[column] = x_test[column].apply(mapper)
x_train.head()
```

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      del sys.path[0]





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
      <th>X290</th>
      <th>X343</th>
      <th>X254</th>
      <th>X141</th>
      <th>X316</th>
      <th>X184</th>
      <th>X263</th>
      <th>X96</th>
      <th>X309</th>
      <th>X32</th>
      <th>...</th>
      <th>X158</th>
      <th>X266</th>
      <th>X192</th>
      <th>X378</th>
      <th>X274</th>
      <th>X56</th>
      <th>X112</th>
      <th>X120</th>
      <th>X164</th>
      <th>X374</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 376 columns</p>
</div>




```python
#Make sure the data is now changed into numericals

print('Feature types:')
x_train[cols].dtypes.value_counts()
```

    Feature types:





    int64    376
    dtype: int64




```python
# Step 4: Perform dimensionality reduction
# Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
n_comp = 12
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(x_train)
pca2_results_test = pca.transform(x_test)
```


```python
# Training using xgboost

import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(pca2_results_train,
                                                      y_train,test_size=0.2,random_state=4242)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
#d_test = xgb.DMatrix(x_test)
d_test = xgb.DMatrix(pca2_results_test)

params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train,1000, watchlist, early_stopping_rounds=50, 
                feval=xgb_r2_score, maximize=True, verbose_eval=10)
```

    [13:53:14] WARNING: /workspace/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	train-rmse:99.14835	valid-rmse:98.26297	train-r2:-58.35295	valid-r2:-67.63754
    Multiple eval metrics have been passed: 'valid-r2' will be used for early stopping.
    
    Will train until valid-r2 hasn't improved in 50 rounds.
    [10]	train-rmse:81.27653	valid-rmse:80.36433	train-r2:-38.88428	valid-r2:-44.91014
    [20]	train-rmse:66.71610	valid-rmse:65.77334	train-r2:-25.87403	valid-r2:-29.75260
    [30]	train-rmse:54.86957	valid-rmse:53.88974	train-r2:-17.17752	valid-r2:-19.64401
    [40]	train-rmse:45.24491	valid-rmse:44.21970	train-r2:-11.35979	valid-r2:-12.89996
    [50]	train-rmse:37.44729	valid-rmse:36.37237	train-r2:-7.46666	valid-r2:-8.40428
    [60]	train-rmse:31.14750	valid-rmse:30.01872	train-r2:-4.85757	valid-r2:-5.40569
    [70]	train-rmse:26.08664	valid-rmse:24.90882	train-r2:-3.10873	valid-r2:-3.41050
    [80]	train-rmse:22.04642	valid-rmse:20.83260	train-r2:-1.93459	valid-r2:-2.08510
    [90]	train-rmse:18.84407	valid-rmse:17.60518	train-r2:-1.14398	valid-r2:-1.20325
    [100]	train-rmse:16.33636	valid-rmse:15.09282	train-r2:-0.61132	valid-r2:-0.61928
    [110]	train-rmse:14.40296	valid-rmse:13.16115	train-r2:-0.25249	valid-r2:-0.23132
    [120]	train-rmse:12.92580	valid-rmse:11.70341	train-r2:-0.00876	valid-r2:0.02634
    [130]	train-rmse:11.81056	valid-rmse:10.63264	train-r2:0.15781	valid-r2:0.19636
    [140]	train-rmse:10.98272	valid-rmse:9.86799	train-r2:0.27173	valid-r2:0.30779
    [150]	train-rmse:10.37499	valid-rmse:9.33268	train-r2:0.35010	valid-r2:0.38085
    [160]	train-rmse:9.92458	valid-rmse:8.97391	train-r2:0.40530	valid-r2:0.42754
    [170]	train-rmse:9.59075	valid-rmse:8.73939	train-r2:0.44464	valid-r2:0.45707
    [180]	train-rmse:9.34082	valid-rmse:8.57620	train-r2:0.47321	valid-r2:0.47716
    [190]	train-rmse:9.15465	valid-rmse:8.47457	train-r2:0.49400	valid-r2:0.48947
    [200]	train-rmse:9.01492	valid-rmse:8.41015	train-r2:0.50932	valid-r2:0.49721
    [210]	train-rmse:8.90950	valid-rmse:8.37241	train-r2:0.52073	valid-r2:0.50171
    [220]	train-rmse:8.82741	valid-rmse:8.34275	train-r2:0.52952	valid-r2:0.50523
    [230]	train-rmse:8.76810	valid-rmse:8.33397	train-r2:0.53582	valid-r2:0.50627
    [240]	train-rmse:8.72179	valid-rmse:8.32743	train-r2:0.54071	valid-r2:0.50705
    [250]	train-rmse:8.67901	valid-rmse:8.32293	train-r2:0.54521	valid-r2:0.50758
    [260]	train-rmse:8.64570	valid-rmse:8.32021	train-r2:0.54869	valid-r2:0.50790
    [270]	train-rmse:8.61560	valid-rmse:8.32217	train-r2:0.55183	valid-r2:0.50767
    [280]	train-rmse:8.58936	valid-rmse:8.32346	train-r2:0.55456	valid-r2:0.50752
    [290]	train-rmse:8.56530	valid-rmse:8.32606	train-r2:0.55705	valid-r2:0.50721
    [300]	train-rmse:8.53588	valid-rmse:8.32534	train-r2:0.56009	valid-r2:0.50730
    [310]	train-rmse:8.50654	valid-rmse:8.32056	train-r2:0.56310	valid-r2:0.50786
    Stopping. Best iteration:
    [260]	train-rmse:8.64570	valid-rmse:8.32021	train-r2:0.54869	valid-r2:0.50790
    



```python
#  Step 5: Predict your test_df values using xgboost

p_test = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = p_test
sub.to_csv('xgb.csv', index=False)

sub.head()
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
      <th>ID</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>82.892540</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>97.114845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>83.577042</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>77.176231</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>111.989700</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
