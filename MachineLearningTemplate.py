#!/usr/bin/env python
# coding: utf-8

# ## Level1 Kaggle ML

# In[1]:


import pandas as pd


# In[42]:


# save filepath to variable for easier access
melbourne_file_path = './data/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
# melbourne_data.describe()


# In[3]:


# print the list of columns
# melbourne_data.columns


# ## Handle String Type Features

# In[43]:


# Count Number of Unique Values a column or feature can have
typ = melbourne_data.dtypes
i = 0
cols_to_drop = []
for col in melbourne_data.columns:
    unq = melbourne_data[col].nunique()
    if typ[i]=='object' and unq>100:
        cols_to_drop.append(col)
#     print(col, unq, typ[i])
    i += 1
print(cols_to_drop)

melbourne_data = melbourne_data.drop(cols_to_drop, axis=1)


# In[44]:


# melbourne_data.head(20)
# one hot encode values
melbourne_data = pd.get_dummies(melbourne_data, drop_first=True)
# melbourne_data.columns
# melbourne_data.head(20)
print(type(melbourne_data))


# ## Impute NA values

# In[74]:


# missing_val_count_by_column = (melbourne_data.isnull().sum())
# print(missing_val_count_by_column)
# print(melbourne_data.dtypes)


# In[45]:


# Impute Values
melbourne_data.fillna(melbourne_data.mean(), inplace=True)
# from sklearn.preprocessing import Imputer
# my_imputer = Imputer()
# melbourne_data = my_imputer.fit_transform(melbourne_data)
print(type(melbourne_data))


# In[16]:


# melbourne_data.dropna(axis=0)
# print(melbourne_data.isnull().sum())


# In[17]:


# our target variable
y = melbourne_data.Price


# In[18]:


# list of features that we want for our model
# melbourne_features =


# In[19]:


# our input data
X = melbourne_data.drop(['Price'], axis=1)


# In[20]:


X.describe()


# In[21]:


X.head()


# In[22]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[23]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model on training data
melbourne_model.fit(train_X, train_y)


# In[24]:


# make prediction on validation data
val_predictions = melbourne_model.predict(val_X)


# In[28]:


from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(val_predictions, val_y))


# In[29]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[30]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[31]:


from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# ## Level 2 Kaggle ML

# ## Using XGBoost boosts our prediction accuracy

# In[32]:


from xgboost import XGBRegressor
xg_model = XGBRegressor(n_estimators=5000, learning_rate=0.01, max_depth=8)
xg_model.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(val_X, val_y)], verbose=False)
xg_pred = xg_model.predict(val_X)
print(mean_absolute_error(val_y, xg_pred))


# ### Partial Dependence Plots

# In[33]:


from sklearn.ensemble import GradientBoostingRegressor

# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(train_X, train_y)


# In[35]:


from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
# Here we make the plot
my_plots = plot_partial_dependence(my_model,       
                                   features=[0], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Rooms', 'Landsize', 'Regionname'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis


# In[36]:


gb_pred = my_model.predict(val_X)
print(mean_absolute_error(val_y, gb_pred))


# ## Using Pipelines

# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=1))


# In[54]:


my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(val_X)


# In[55]:


print(mean_absolute_error(predictions, val_y))


# ## Using Cross Validation

# In[57]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)
print(scores)


# In[ ]:




