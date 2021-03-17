#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries:-

# In[1]:


import pandas as pd
import numpy as np
import random


# # Reading CSV File:-

# In[2]:


X = pd.read_csv("data.csv")
X.head()


# In[3]:


columns = X.columns # get columns name
columns


# # Min-Max Normalization:-

# In[88]:


def normalize(df):
    normalized = df.copy()
    for features in columns:
        max_value = df[features].max()
        min_value = df[features].min()
        normalized[features] = (df[features] - min_value) / (max_value - min_value)
    return normalized


# In[89]:


normalize_X = normalize(X)
normalize_X.to_csv("data_scaled.csv")
normalize_X.head()


# # Randomly Creating Missingness:-

# In[6]:


def missing_data(dataframe, normalize_dataframe):
    df1 = dataframe.copy()
    df2 = normalize_dataframe.copy()
    
    features = columns
    
    for col in dataframe[features]:
        df_split = dataframe.sample(frac=0.5,random_state=200)  # select 50% rows randomly
        df_split.reset_index()
        indices = df_split.sample(frac=0.5, replace=True).index # take random rows of those 50%
        df1.loc[indices,col] = np.nan # set nan at this positions
        df2.loc[indices, col] = np.nan
    return df1, df2                                                              


# In[7]:


X_original_missing,normalize_X_missing = missing_data(X, normalize_X)
X_original_missing.head()


# In[8]:


normalize_X_missing.head()


# In[53]:


'''
dataframe convert into array.
'''
ori_data = X_original_missing.values
normdata = normalize_X_missing.values


# In[54]:


attributes = ori_data.shape[1] # 7 cols
instances = ori_data.shape[0] # 101 rows


# In[55]:


null_indexes = np.argwhere(np.isnan(ori_data)) # row_col combinations who has null values


# # Imputation:-

# ## Mean Method:-

# ### mean imputation function

# In[12]:


def mean(dataframe):
    features = columns
    for col in features:
        avg = dataframe[col].mean()
        dataframe[col].replace(np.nan, avg, inplace = True)
    return dataframe


# #### ==> Original dataframe's imputation with mean

# In[13]:


X_mean_impute = X_original_missing.copy()
mean(X_mean_impute) # mean imputation
X_mean_impute.head()


# #### ==> Scaled dataframe's imputation with mean

# In[14]:


normalize_X_mean_impute = normalize_X_missing.copy()
mean(normalize_X_mean_impute) # normalize-mean imputation
normalize_X_mean_impute.head()


# ### Calculating Euclidean Distance For KNN:-

# In[62]:


def euclidean_distance(i,j): 
    column = len(i)
    distance = 0
    for a in range(column):
        if np.isnan(i[a]): # if there are nan in missing index then skip it
            continue
        distance = distance + (i[a]-j[a])**2
    return np.sqrt(distance)


# ### KNN:-
# 

# In[98]:


'''
Reference of KNN :- https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

'''

def sortSecond(val): 
    return val[1] # get value of 1st position

def knn(missingdata,k):
    fill_value = missingdata.copy()
    for j in null_indexes: # null_indexes contains all the row_col combinations of missing value
        distances = []
        for i in range(instances):
            if(i==j[0]): # if two rows are same then skip it
                continue
            else:
                if np.isnan(missingdata[i]).any(): # if row contains null anywhere then it skips
                    continue
                d = euclidean_distance(missingdata[j[0]], missingdata[i])  # distance only calculate for complete case
                distances.append([missingdata[i][j[1]], d] ) 
        distances.sort(key = sortSecond) # call sortSecond function and sort the list on the basis of value of 1st position in this case it is "distance"
        distances = np.array(distances)
        count = len(distances)
        nearest = [] # store the nearest neighbor in list
        for l in range(count):
            nearest.append(distances[l,0])    
        x = nearest[:k] #  slice list on the value of k
        mean = sum(x) / len(x)
        fill_value[j[0]][j[1]] = mean  # replace the nan with mean value  
    return fill_value


# #### ==> Original's dataframe imputation with k=1,3,5

# In[99]:


knn1 = knn(ori_data, 1)
knn1_df = pd.DataFrame(knn1)
knn1_df.columns = columns
knn1_df.head(10)


# In[57]:


knn3 = knn(ori_data, 3)
knn3_df = pd.DataFrame(knn3)
knn3_df.columns = columns
knn3_df.head()


# In[58]:


knn5 = knn(ori_data, 5)
knn5_df = pd.DataFrame(knn5)
knn5_df.columns = columns
knn5_df.head()


# #### ==> Scaled dataframe imputation with k=1,3,5

# In[20]:


scaled_knn1 = knn(normdata, 1)
scaled_knn1_df = pd.DataFrame(scaled_knn1)
scaled_knn1_df.columns = columns
scaled_knn1_df.head()


# In[21]:


scaled_knn3 = knn(normdata, 3)
scaled_knn3_df = pd.DataFrame(scaled_knn3)
scaled_knn3_df.columns = columns
scaled_knn3_df.head()


# In[22]:


scaled_knn5 = knn(normdata, 5)
scaled_knn5_df = pd.DataFrame(scaled_knn5)
scaled_knn5_df.columns = columns
scaled_knn5_df.head()


# ### Calculating Euclidean Distance For Weighted KNN:-

# In[100]:


def weighted_euclidean_distance(i,j):
    attr = len(i)
    distance = 0
    for k in range(attr):
        if np.isnan(i[k]):
            continue
        distance = distance + np.square(i[k]-j[k])    
    return np.sqrt(distance)*(1-distance)


# ### Weighted KNN:-

# In[101]:


def sortSecond(val): 
    return val[1] # get value of 1st position

def weight_knn(missingdata,k):
    fill_value = missingdata.copy()
    for j in null_indexes:  # null_indexes contains all the row_col combinations of missing value
        distances = [] 
        for i in range(instances):
            if(i==j[0]): # if two rows are same then skip it
                continue
            else:
                if np.isnan(missingdata[i]).any(): # if row contains null anywhere then it skips
                    continue
                d = weighted_euclidean_distance(missingdata[j[0]], missingdata[i]) # distance only calculate for complete case
                distances.append([missingdata[i][j[1]], d])
        distances.sort(key = sortSecond) # call sortSecond function and sort the list on the basis of value of 1st position in this case it is "distance"
        distances = np.array(distances)
        count = len(distances)
        nearest = [] # store the nearest neighbor in list
        for l in range(count):
            nearest.append(distances[l,0])    
        x = nearest[:k] # slice list on the value of k
        mean = sum(x) / len(x)
        fill_value[j[0]][j[1]] = mean    # replace the nan with mean value
    return fill_value


# #### ==> Original's dataframe imputation with weighted k=1,3,5

# In[102]:


weight_knn1 = weight_knn(ori_data, 1)
weight_knn1_df = pd.DataFrame(weight_knn1)
weight_knn1_df.columns = columns
weight_knn1_df.head()


# In[60]:


weight_knn3 = weight_knn(ori_data, 3)
weight_knn3_df = pd.DataFrame(weight_knn3)
weight_knn3_df.columns = columns
weight_knn3_df.head()


# In[61]:


weight_knn5 = weight_knn(ori_data, 5)
weight_knn5_df = pd.DataFrame(weight_knn5)
weight_knn5_df.columns = columns
weight_knn5_df.head()


# #### Scaled dataframe imputation with weighted k=1,3,5

# In[28]:


weight_scaled_knn1 = weight_knn(normdata, 1)
weight_scaled_knn1_df = pd.DataFrame(weight_scaled_knn1)
weight_scaled_knn1_df.columns = columns
weight_scaled_knn1_df.head()


# In[29]:


weight_scaled_knn3 = weight_knn(normdata, 3)
weight_scaled_knn3_df = pd.DataFrame(weight_scaled_knn3)
weight_scaled_knn3_df.columns = columns
weight_scaled_knn3_df.head()


# In[30]:


weight_scaled_knn5 = weight_knn(normdata, 5)
weight_scaled_knn5_df = pd.DataFrame(weight_scaled_knn5)
weight_scaled_knn5_df.columns = columns
weight_scaled_knn5_df.head()


# # MSE:-

# In[31]:


def mean_square_error_calculation(x, y):
    total = 0
    for row in range(len(x)):
        error = np.square(x[row] - y[row])
        total += error
    mse = total/len(x)
    return mse

def MSE(actual_df, impute_df):
    mse = []
    for i in columns:
        cal = mean_square_error_calculation(actual_df[i], impute_df[i])
        mse.append(cal)
    return mse


# #### MSE for mean imputation original data:-

# In[32]:


actual_mse_mean = MSE(X, X_mean_impute)


# #### MSE for k=1,3,5 imputation original data:-

# In[33]:


actual_mse_knn1 = MSE(X, knn1_df)


# In[34]:


actual_mse_knn3 = MSE(X, knn3_df)


# In[35]:


actual_mse_knn5 = MSE(X, knn5_df)


# #### MSE for weighted k=1,3,5 imputation original data:-

# In[36]:


actual_mse_Wknn1 = MSE(X, weight_knn1_df)


# In[37]:


actual_mse_Wknn3 = MSE(X, weight_knn3_df)


# In[38]:


actual_mse_Wknn5 = MSE(X, weight_knn5_df)


# In[39]:


col = ["Mean", "KNN-1", "KNN-3", "KNN-5", "WEIGHTED-KNN-1", "WEIGHTED-KNN-3", "WEIGHTED-KNN-5"]
results_original_data = pd.DataFrame([actual_mse_mean, actual_mse_knn1, actual_mse_knn3, actual_mse_knn5, actual_mse_Wknn1, actual_mse_Wknn3, actual_mse_Wknn5])
results_original_data = results_original_data.transpose()
results_original_data.columns = col
results_original_data.index = columns
results_original_data.head(7)


# In[40]:


results_original_data.to_csv("results_original_data.csv")


# #### MSE for mean imputation normalize data:-

# In[41]:


normalize_mse_mean = MSE(normalize_X, normalize_X_mean_impute)


# #### MSE for k=1,3,5 imputation normalize data:-

# In[42]:


normalize_mse_knn1 = MSE(normalize_X, scaled_knn1_df)


# In[43]:


normalize_mse_knn3 = MSE(normalize_X, scaled_knn3_df)


# In[44]:


normalize_mse_knn5 = MSE(normalize_X, scaled_knn5_df)


# #### MSE for weighted k=1,3,5 imputation normalize data:-

# In[45]:


normalize_mse_Wknn1 = MSE(normalize_X, weight_scaled_knn1_df)


# In[46]:


normalize_mse_Wknn3 = MSE(normalize_X, weight_scaled_knn3_df)


# In[47]:


normalize_mse_Wknn5 = MSE(normalize_X, weight_scaled_knn5_df)


# In[48]:


norm_col = ["Mean", "KNN-1", "KNN-3", "KNN-5", "WEIGHTED-KNN-1", "WEIGHTED-KNN-3", "WEIGHTED-KNN-5"]
results_scaled_data = pd.DataFrame([normalize_mse_mean, normalize_mse_knn1, normalize_mse_knn3, normalize_mse_knn5, normalize_mse_Wknn1, normalize_mse_Wknn3, normalize_mse_Wknn5])
results_scaled_data = results_scaled_data.transpose()
results_scaled_data.columns = norm_col
results_scaled_data.index = columns
results_scaled_data.head(7)


# In[49]:


results_scaled_data.to_csv("results_scaled_data.csv")

