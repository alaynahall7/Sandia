import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.path as mpath
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from neuralprophet import NeuralProphet, set_log_level


#region PCA

#import the data

#og = original

og_temp = pd.read_csv('/Users/rachelalaynahall/Desktop/NP/.conda/split_og_ttemp.csv')
og_aod = pd.read_csv('/Users/rachelalaynahall/Desktop/NP/.conda/split_og_taod.csv')

#add name to date column
og_temp.rename( columns={'Unnamed: 0':'Date'}, inplace=True )
og_aod.rename( columns={'Unnamed: 0':'Date'}, inplace=True )

#convert to df
og_temp = pd.DataFrame(og_temp)
og_aod = pd.DataFrame(og_aod)

#remove ".X" from column headings
og_temp.columns = og_temp.columns.str.strip('X.')
og_aod.columns = og_aod.columns.str.strip('X.')

#convert date to datetime
og_temp['Date'] = pd.to_datetime(og_temp['Date'])
og_aod['Date'] = pd.to_datetime(og_aod['Date'])

#separate train and test data

#ot = original temp

ot_train = og_temp[og_temp['Date'] < '1991-01-01'] 
ot_test = og_temp[og_temp['Date'] > '1990-12-31']

#oa = original aod

oa_train = og_aod[og_aod['Date'] < '1991-01-01'] 
oa_test = og_aod[og_aod['Date'] > '1990-12-31']

#set date as index
ot_train = ot_train.set_index('Date')
oa_train = oa_train.set_index('Date')

ot_test = ot_test.set_index('Date')
oa_test = oa_test.set_index('Date')

#create features list
ot_test_list = list(ot_test)
features = ot_test_list

#create train and test variables of features
x_ot_train = ot_train.loc[:, features].values
x_ot_test = ot_test.loc[:, features].values
x_oa_train = oa_train.loc[:, features].values
x_oa_test = oa_test.loc[:, features].values

#create scaler functions
scaler_ot = StandardScaler()
scaler_oa = StandardScaler()

#fit scalers on training data
scaler_ot.fit(x_ot_train)
scaler_oa.fit(x_oa_train)

#scale data
x_ot_train = scaler_ot.transform(x_ot_train)
x_oa_train = scaler_oa.transform(x_oa_train)

x_ot_test = scaler_ot.transform(x_ot_test)
x_oa_test = scaler_oa.transform(x_oa_test)

#create pca functions
pca_ot = PCA(n_components = 5)
pca_oa = PCA(n_components = 5)

#fit pca functions on training data
pca_ot.fit(x_ot_train)
pca_oa.fit(x_oa_train)

#transform data through pca
principalComponents_ot_train = pca_ot.transform(x_ot_train)
principalComponents_oa_train = pca_oa.transform(x_oa_train)

principalComponents_ot_test = pca_ot.transform(x_ot_test)
principalComponents_oa_test = pca_oa.transform(x_oa_test)

#create new dfs with principal components
principalDf_ot_train = pd.DataFrame(data = principalComponents_ot_train
             , columns = ['principal component 1 - Temp', 'principal component 2 - Temp', 'principal component 3 - Temp', 'principal component 4 - Temp','principal component 5 - Temp'])
principalDf_oa_train = pd.DataFrame(data = principalComponents_oa_train
             , columns = ['principal component 1 - AOD', 'principal component 2 - AOD', 'principal component 3 - AOD', 'principal component 4 - AOD','principal component 5 - AOD'])

principalDf_ot_test = pd.DataFrame(data = principalComponents_ot_test
             , columns = ['principal component 1 - Temp', 'principal component 2 - Temp', 'principal component 3 - Temp', 'principal component 4 - Temp','principal component 5 - Temp'])
principalDf_oa_test = pd.DataFrame(data = principalComponents_oa_test
             , columns = ['principal component 1 - AOD', 'principal component 2 - AOD', 'principal component 3 - AOD', 'principal component 4 - AOD','principal component 5 - AOD'])

#add back in date column
principalDf_ot_train['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ot_train), freq='D')
principalDf_oa_train['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_oa_train), freq='D')

principalDf_ot_test['date'] = pd.date_range(start='1/1/1991', periods = len(principalDf_ot_test), freq='D')
principalDf_oa_test['date'] = pd.date_range(start='1/1/1991', periods = len(principalDf_oa_test), freq='D')

#set dates as index
#principalDf_ot_train = principalDf_ot_train.set_index('date')
#principalDf_oa_train = principalDf_oa_train.set_index('date')

#principalDf_ot_test = principalDf_ot_test.set_index('date')
#principalDf_oa_test = principalDf_oa_test.set_index('date')

#view explained variance
pca_ot.explained_variance_ratio_
pca_oa.explained_variance_ratio_

#view dfs
principalDf_ot_train
principalDf_oa_train

principalDf_ot_test
principalDf_oa_test

#convert PCA back to original space
ot_orig = np.dot(principalComponents_ot_test,pca_ot.components_)
ot_orig_backscaled = scaler_ot.inverse_transform(ot_orig)

ot_orig = pd.DataFrame(ot_orig)
ot_orig_backscaled = pd.DataFrame(ot_orig_backscaled)

ot_train.head()
principalDf_ot_train.head()

ot_orig.head()
ot_orig_backscaled.head()


oa_orig = np.dot(principalComponents_oa_test,pca_oa.components_)
oa_orig_backscaled = scaler_oa.inverse_transform(oa_orig)

oa_orig = pd.DataFrame(oa_orig)
oa_orig_backscaled = pd.DataFrame(oa_orig_backscaled)

oa_train.head()
principalDf_oa_train.head()

oa_orig.head()
oa_orig_backscaled.head()

#endregion 

#region Split Test/Train

# Load the dataset from the CSV file using pandas
df1_test = principalDf_ot_test['date'].copy()
temp1 = principalDf_ot_test['principal component 1 - Temp'].copy()
aod1 = principalDf_oa_test['principal component 1 - AOD'].copy()
df1_test = pd.concat([df1_test, temp1], axis = 1)
df1_test = pd.concat([df1_test, aod1], axis = 1)

df2_test = principalDf_ot_test['date'].copy()
temp2 = principalDf_ot_test[["principal component 2 - Temp"]].copy()
aod2 = principalDf_oa_test[["principal component 2 - AOD"]].copy()
df2_test = pd.concat([df2_test, temp2], axis = 1)
df2_test = pd.concat([df2_test, aod2], axis = 1)

df3_test = principalDf_ot_test['date'].copy()
temp3 = principalDf_ot_test[["principal component 3 - Temp"]].copy()
aod3 = principalDf_oa_test[["principal component 3 - AOD"]].copy()
df3_test = pd.concat([df3_test, temp3], axis = 1)
df3_test = pd.concat([df3_test, aod3], axis = 1)

df4_test = principalDf_ot_test['date'].copy()
temp4 = principalDf_ot_test[["principal component 4 - Temp"]] 
aod4 = principalDf_oa_test[["principal component 4 - AOD"]].copy()
df4_test = pd.concat([df4_test, temp4], axis = 1)
df4_test = pd.concat([df4_test, aod4], axis = 1)

df5_test = principalDf_ot_test['date'].copy()
temp5= principalDf_ot_test[["principal component 5 - Temp"]].copy()
aod5 = principalDf_oa_test[["principal component 5 - AOD"]].copy()
df5_test = pd.concat([df5_test, temp5], axis = 1)
df5_test = pd.concat([df5_test, aod5], axis = 1)

df1_train = principalDf_ot_train['date'].copy()
temp1a = principalDf_ot_train['principal component 1 - Temp'].copy()
aod1a = principalDf_oa_train['principal component 1 - AOD'].copy()
df1_train = pd.concat([df1_train, temp1a], axis = 1)
df1_train = pd.concat([df1_train, aod1a], axis = 1)

df2_train = principalDf_ot_train['date'].copy()
temp2a = principalDf_ot_train[["principal component 2 - Temp"]].copy()
aod2a = principalDf_oa_train[["principal component 2 - AOD"]].copy()
df2_train = pd.concat([df2_train, temp2a], axis = 1)
df2_train = pd.concat([df2_train, aod2a], axis = 1)

df3_train = principalDf_ot_train['date'].copy()
temp3a = principalDf_ot_train[["principal component 3 - Temp"]].copy()
aod3a = principalDf_oa_train[["principal component 3 - AOD"]].copy()
df3_train = pd.concat([df3_train, temp3a], axis = 1)
df3_train = pd.concat([df3_train, aod3a], axis = 1)

df4_train = principalDf_ot_train['date'].copy()
temp4a = principalDf_ot_train[["principal component 4 - Temp"]].copy()
aod4a = principalDf_oa_train[["principal component 4 - AOD"]].copy()
df4_train = pd.concat([df4_train, temp4a], axis = 1)
df4_train = pd.concat([df4_train, aod4a], axis = 1)

df5_train = principalDf_ot_train['date'].copy()
temp5a = principalDf_ot_train[["principal component 5 - Temp"]].copy()
aod5a = principalDf_oa_train[["principal component 5 - AOD"]].copy()
df5_train = pd.concat([df5_train, temp5a], axis = 1)
df5_train = pd.concat([df5_train, aod5a], axis = 1)

df1_train.rename(columns = {'date':'ds'}, inplace = True)
df2_train.rename(columns = {'date':'ds'}, inplace = True)
df3_train.rename(columns = {'date':'ds'}, inplace = True)
df4_train.rename(columns = {'date':'ds'}, inplace = True)
df5_train.rename(columns = {'date':'ds'}, inplace = True)

df1_test.rename(columns = {'date':'ds'}, inplace = True)
df2_test.rename(columns = {'date':'ds'}, inplace = True)
df3_test.rename(columns = {'date':'ds'}, inplace = True)
df4_test.rename(columns = {'date':'ds'}, inplace = True)
df5_test.rename(columns = {'date':'ds'}, inplace = True)

df1_train.rename(columns = {'principal component 1 - Temp':'y'}, inplace = True)
df2_train.rename(columns = {'principal component 2 - Temp':'y'}, inplace = True)
df3_train.rename(columns = {'principal component 3 - Temp':'y'}, inplace = True)
df4_train.rename(columns = {'principal component 4 - Temp':'y'}, inplace = True)
df5_train.rename(columns = {'principal component 5 - Temp':'y'}, inplace = True)

df1_test.rename(columns = {'principal component 1 - Temp':'y'}, inplace = True)
df2_test.rename(columns = {'principal component 2 - Temp':'y'}, inplace = True)
df3_test.rename(columns = {'principal component 3 - Temp':'y'}, inplace = True)
df4_test.rename(columns = {'principal component 4 - Temp':'y'}, inplace = True)
df5_test.rename(columns = {'principal component 5 - Temp':'y'}, inplace = True)

df1_train.rename(columns = {'principal component 1 - AOD':'x'}, inplace = True)
df2_train.rename(columns = {'principal component 2 - AOD':'x'}, inplace = True)
df3_train.rename(columns = {'principal component 3 - AOD':'x'}, inplace = True)
df4_train.rename(columns = {'principal component 4 - AOD':'x'}, inplace = True)
df5_train.rename(columns = {'principal component 5 - AOD':'x'}, inplace = True)

df1_test.rename(columns = {'principal component 1 - AOD':'x'}, inplace = True)
df2_test.rename(columns = {'principal component 2 - AOD':'x'}, inplace = True)
df3_test.rename(columns = {'principal component 3 - AOD':'x'}, inplace = True)
df4_test.rename(columns = {'principal component 4 - AOD':'x'}, inplace = True)
df5_test.rename(columns = {'principal component 5 - AOD':'x'}, inplace = True)


df1 = df1_train
df1 = pd.concat([df1, df1_test], axis = 0)

df2 = df2_train
df2 = pd.concat([df2, df2_test], axis = 0)

df3 = df3_train
df3 = pd.concat([df3, df3_test], axis = 0)

df4 = df4_train
df4 = pd.concat([df4, df4_test], axis = 0)

df5 = df5_train
df5 = pd.concat([df5, df5_test], axis = 0)


# Disable logging messages unless there is an error
set_log_level("ERROR")

print("Dataset size:", len(df1))
print("Train dataset size:", len(df1_train))
print("Test dataset size:", len(df1_test))

print("Dataset size:", len(df2))
print("Train dataset size:", len(df2_train))
print("Test dataset size:", len(df2_test))

print("Dataset size:", len(df3))
print("Train dataset size:", len(df3_train))
print("Test dataset size:", len(df3_test))

print("Dataset size:", len(df4))
print("Train dataset size:", len(df4_train))
print("Test dataset size:", len(df4_test))

print("Dataset size:", len(df5))
print("Train dataset size:", len(df5_train))
print("Test dataset size:", len(df5_test))

df1_nx = df1['ds'].copy()
nx = df1['y'].copy()
df1_nx = pd.concat([df1_nx, nx], axis = 1)

#endregion

confidence_level = 0.95

boundaries = round((1 - confidence_level) / 2, 2)
# NeuralProphet only accepts quantiles value in between 0 and 1
quantiles = [boundaries, confidence_level + boundaries]

# Model and prediction
m = NeuralProphet()
m.set_plotting_backend("matplotlib")

m.add_lagged_regressor('x')

m.fit(df1_train)

forecast = m.predict(df1)
m.plot(forecast)

df1t_test = df1_test.drop(columns = ['x'])
df1t_train = df1_train.drop(columns = ['x'])
df1t = df1.drop(columns = ['x'])

df2t_test = df2_test.drop(columns = ['x'])
df2t_train = df2_train.drop(columns = ['x'])
df2t = df2.drop(columns = ['x'])

df3t_test = df3_test.drop(columns = ['x'])
df3t_train = df3_train.drop(columns = ['x'])
df3t = df3.drop(columns = ['x'])

df4t_test = df4_test.drop(columns = ['x'])
df4t_train = df4_train.drop(columns = ['x'])
df4t = df4.drop(columns = ['x'])

df5t_test = df5_test.drop(columns = ['x'])
df5t_train = df5_train.drop(columns = ['x'])
df5t = df5.drop(columns = ['x'])

m1 = NeuralProphet(
    quantiles=quantiles,
)
m1.set_plotting_backend("matplotlib")

m1t = NeuralProphet(
    quantiles=quantiles,
)
m1t.set_plotting_backend("matplotlib")

m2t = NeuralProphet(
    quantiles=quantiles,
)
m2t.set_plotting_backend("matplotlib")

m3t = NeuralProphet(
    quantiles=quantiles,
)
m3t.set_plotting_backend("matplotlib")

m4t = NeuralProphet(
    quantiles=quantiles,
)
m4t.set_plotting_backend("matplotlib")

m5t = NeuralProphet(
    quantiles=quantiles,
)
m5t.set_plotting_backend("matplotlib")



metrics1t = m1t.fit(df1t_train, validation_df = df1t_test, progress = None)
metrics1t

forecast1t = m1t.predict(df1t)
m1t.plot(forecast1t)

metrics2t = m2t.fit(df2t_train, validation_df = df2t_test, progress = None)
metrics2t

forecast2t = m2t.predict(df2t)
m2t.plot(forecast2t)

metrics3t = m3t.fit(df3t_train, freq = 'D', epochs = 1000, validation_df = df3t_test, progress = None)
metrics3t

forecast3t = m3t.predict(df3t)
m3t.plot(forecast3t)

metrics4t = m4t.fit(df4t_train, freq = 'D', epochs = 1000, validation_df = df4t_test, progress = None)
metrics4t

forecast4t = m4t.predict(df4t)
m4t.plot(forecast4t)

metrics5t = m5t.fit(df5t_train, freq = 'D', epochs = 1000, validation_df = df5t_test, progress = None)
metrics5t

forecast5t = m5t.predict(df5t)
m5t.plot(forecast5t)
