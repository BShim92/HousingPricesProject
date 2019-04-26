


import os
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')



dropcolumns = 225

inputs = pd.read_csv('traincleaned.csv')
testinputs = pd.read_csv('testcleaned.csv')
featureimportance = pd.read_csv('feature importance.csv')
print(featureimportance.shape)
featureimportance = featureimportance.rename(columns={'0': 'importance', '1':'feature'})
featureimportance = featureimportance.drop(columns=['Unnamed: 0'])
featureimportance = featureimportance.sort_values(by='importance')
featureimportance = featureimportance[0:dropcolumns]
featureimportance = featureimportance['feature'].tolist()

inputs = inputs.drop(columns=['SalePrice'])
alldata = pd.concat([inputs, testinputs])
alldata.set_index('Id', inplace=True)
alldata = alldata.fillna(0)
alldata = pd.get_dummies(alldata)
alldata = alldata.drop(columns=['Unnamed: 0'])
print(alldata.shape)
for column in featureimportance:
    try:
        alldata = alldata.drop(columns=column)
    except:
        print(f'Couldnt drop column {column}')

print(alldata.shape)

inputs = alldata.loc[0:1460,]
testinputs = alldata.loc[1461:]

numericalcolumns = []
for column in inputs.columns:
    if set(inputs[column].tolist()) != {0, 1}:
        numericalcolumns.append(column)


prices = pd.read_csv('traincleaned.csv')
prices = prices['SalePrice']
prices = np.array(prices)
if tolog == 'Yes':
    prices = np.log10(prices)

testinputs = testinputs.values
inputs = inputs.values



from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Fitting our model with all of our features in X
model.fit(inputs, prices)

score = model.score(inputs, prices)

predictions = model.predict(inputs)
predictions = np.power(10, predictions)
pricesunlogged = np.power(10, prices)

RMSLE = math.sqrt(mean_squared_log_error(pricesunlogged, predictions))

predictions = model.predict(testinputs)
predictions = np.power(10, predictions)

print(f'R2: {score}, RMSLE: {RMSLE}, dropcolumns: {dropcolumns}')
#0.12363, R2 Score = .94, RMSLE = .0977, no scaling, drop columns = 150
#.12054, R2: 0.9361934187245556, RMSLE: 0.10093248123296446, dropcolumns: 200
#.12057, R2: 0.9299809808208424, RMSLE: 0.10573194685650467, dropcolumns: 225, with all feature engineering




from sklearn.linear_model import Lasso

for alpha in range(1):
    alpha = alpha/10
    lasso = Lasso(alpha=alpha).fit(inputs, prices)

    predictions = lasso.predict(inputs)
    predictions = np.power(10, predictions)
    pricesunlogged = np.power(10, prices)


    RMSLE = math.sqrt(mean_squared_log_error(pricesunlogged, predictions))

    r2 = lasso.score(inputs, prices)

    predictions = lasso.predict(testinputs)
    predictions = np.power(10, predictions)

    print(f"alpha: {alpha}, RMSLE: {RMSLE}, R2: {r2}, dropcolumns: {dropcolumns}")
#.11974, alpha: 0.0, RMSLE: 0.1062362486329152, R2: 0.9293114626723123, dropcolumns: 225  


# In[53]:


from sklearn.linear_model import Ridge

for alpha in range(10):
    alpha = alpha/10
    ridge = Ridge(alpha=alpha).fit(inputs, prices)

    predictions = ridge.predict(inputs)
    predictions = np.power(10, predictions)
    pricesunlogged = np.power(10, prices)


    RMSLE = math.sqrt(mean_squared_log_error(pricesunlogged, predictions))

    r2 = ridge.score(inputs, prices)

    predictions = ridge.predict(testinputs)
    predictions = np.power(10, predictions)

    print(f"alpha: {alpha}, RMSLE: {RMSLE}, R2: {r2}, dropcolumns: {dropcolumns}")


# In[54]:


from sklearn.linear_model import ElasticNet

for alpha in range(10):
    alpha = alpha/10
    elasticnet = ElasticNet(alpha=alpha).fit(inputs, prices)

    predictions = elasticnet.predict(inputs)
    predictions = np.power(10, predictions)
    pricesunlogged = np.power(10, prices)


    RMSLE = math.sqrt(mean_squared_log_error(pricesunlogged, predictions))

    r2 = elasticnet.score(inputs, prices)

    predictions = elasticnet.predict(testinputs)
    predictions = np.power(10, predictions)

    print(f"alpha: {alpha}, RMSLE: {RMSLE}, R2: {r2}, dropcolumns: {dropcolumns}")


# In[46]:


print(predictions)


# In[ ]:





# In[56]:


submittest = pd.read_csv('testcleaned.csv')
submittest = submittest[['Id']]
submittest['SalePrice'] = predictions
submittest.to_csv('BrandenSubmission.csv', index=False)
print(submittest.shape, len(predictions))
print(f'tolog = {tolog}')
submittest.head()


# In[ ]:




