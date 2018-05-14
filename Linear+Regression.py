
# coding: utf-8

# # Linear Regression 

# In[1]:


import pandas as pd
import seaborn as sb
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')


# In[2]:


#Read CSV file from URL
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv')

#Display the first 5 rows
data.head()


# In[3]:


#Check the shape of the DataFrame (Rows, Columns)
data.shape


# In[7]:


sb.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')


# In[8]:


#Creat a list of feature names. Need to creat a df with features
feature_cols = ['TV', 'radio', 'newspaper']

# Use the list to select a subset of the origional DataFrame
X = data[feature_cols]

# Equivalent command to do this in one line
X = data[['TV', 'radio', 'newspaper']]

# Print first 5 rows
X.head()


# In[12]:


print(type(X))
print(X.shape)


# In[13]:


# Select a series from the df. Select response value 
y = data['sales']

# Command that works if there are not spaces
y = data.sales
# First 5 rows

y.head()


# In[14]:


print(type(y))
print(y.shape)


# In[15]:


# Split x an y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =1)


# In[16]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[18]:


# Import model 
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

# Fit the model to the training data (Learn the coefficients)

linreg.fit(X_train, y_train)


# In[19]:


print(linreg.intercept_)
print(linreg.coef_)

