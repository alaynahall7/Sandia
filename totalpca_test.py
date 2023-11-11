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

ot = og_temp.set_index('Date')
oa = og_aod.set_index('Date')

#create features list
loc_list = list(ot)
features = loc_list

#create train and test variables of features

x_ot = ot.loc[:, features].values
x_oa = oa.loc[:, features].values

#create scaler functions
scaler_ot = StandardScaler()
scaler_oa = StandardScaler()

#fit scalers on training data
scaler_ot.fit(x_ot)
scaler_oa.fit(x_oa)

#scale data
x_ot = scaler_ot.transform(x_ot)
x_oa = scaler_oa.transform(x_oa)

#create pca functions

pca_ot = PCA(n_components = 5)
pca_oa = PCA(n_components = 5)

#fit pca functions on training data

pca_ot.fit(x_ot)
pca_oa.fit(x_oa)

#transform data through pca

principalComponents_ot = pca_ot.transform(x_ot)
principalComponents_oa = pca_oa.transform(x_oa)

#########

#create new dfs with principal components
principalDf_ot = pd.DataFrame(data = principalComponents_ot
             , columns = ['principal component 1 - Temp', 'principal component 2 - Temp', 'principal component 3 - Temp', 'principal component 4 - Temp','principal component 5 - Temp'])
principalDf_oa = pd.DataFrame(data = principalComponents_oa
             , columns = ['principal component 1 - AOD', 'principal component 2 - AOD', 'principal component 3 - AOD', 'principal component 4 - AOD','principal component 5 - AOD'])


#add back in date column
principalDf_ot['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ot), freq='D')
principalDf_oa['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_oa), freq='D')


#view explained variance
pca_ot.explained_variance_ratio_
pca_oa.explained_variance_ratio_

#view dfs
principalDf_ot
principalDf_oa

principalDf_ot['date'] = pd.to_datetime(principalDf_ot['date'])
principalDf_oa['date'] = pd.to_datetime(principalDf_oa['date'])

split_date = pd.datetime(1990,12,31)

ot_train = principalDf_ot.loc[principalDf_ot['date'] <= split_date]
ot_test = principalDf_ot.loc[principalDf_ot['date'] > split_date]

oa_train = principalDf_oa.loc[principalDf_oa['date'] <= split_date]
oa_test = principalDf_oa.loc[principalDf_oa['date'] > split_date]


#endregion



#region Split Train/Test data

df1_test = ot_test['date'].copy()
temp1 = ot_test['principal component 1 - Temp'].copy()
aod1 = oa_test['principal component 1 - AOD'].copy()
df1_test = pd.concat([df1_test, temp1], axis = 1)
df1_test = pd.concat([df1_test, aod1], axis = 1)

df2_test = ot_test['date'].copy()
temp2 = ot_test[["principal component 2 - Temp"]].copy()
aod2 = oa_test[["principal component 2 - AOD"]].copy()
df2_test = pd.concat([df2_test, temp2], axis = 1)
df2_test = pd.concat([df2_test, aod2], axis = 1)

df3_test = ot_test['date'].copy()
temp3 = ot_test[["principal component 3 - Temp"]].copy()
aod3 = oa_test[["principal component 3 - AOD"]].copy()
df3_test = pd.concat([df3_test, temp3], axis = 1)
df3_test = pd.concat([df3_test, aod3], axis = 1)

df4_test = ot_test['date'].copy()
temp4 = ot_test[["principal component 4 - Temp"]] 
aod4 = oa_test[["principal component 4 - AOD"]].copy()
df4_test = pd.concat([df4_test, temp4], axis = 1)
df4_test = pd.concat([df4_test, aod4], axis = 1)

df5_test = ot_test['date'].copy()
temp5= ot_test[["principal component 5 - Temp"]].copy()
aod5 = oa_test[["principal component 5 - AOD"]].copy()
df5_test = pd.concat([df5_test, temp5], axis = 1)
df5_test = pd.concat([df5_test, aod5], axis = 1)

df1_train = ot_train['date'].copy()
temp1a = ot_train['principal component 1 - Temp'].copy()
aod1a = oa_train['principal component 1 - AOD'].copy()
df1_train = pd.concat([df1_train, temp1a], axis = 1)
df1_train = pd.concat([df1_train, aod1a], axis = 1)

df2_train = ot_train['date'].copy()
temp2a = ot_train[["principal component 2 - Temp"]].copy()
aod2a = oa_train[["principal component 2 - AOD"]].copy()
df2_train = pd.concat([df2_train, temp2a], axis = 1)
df2_train = pd.concat([df2_train, aod2a], axis = 1)

df3_train = ot_train['date'].copy()
temp3a = ot_train[["principal component 3 - Temp"]].copy()
aod3a = oa_train[["principal component 3 - AOD"]].copy()
df3_train = pd.concat([df3_train, temp3a], axis = 1)
df3_train = pd.concat([df3_train, aod3a], axis = 1)

df4_train = ot_train['date'].copy()
temp4a = ot_train[["principal component 4 - Temp"]].copy()
aod4a = oa_train[["principal component 4 - AOD"]].copy()
df4_train = pd.concat([df4_train, temp4a], axis = 1)
df4_train = pd.concat([df4_train, aod4a], axis = 1)

df5_train = ot_train['date'].copy()
temp5a = ot_train[["principal component 5 - Temp"]].copy()
aod5a = oa_train[["principal component 5 - AOD"]].copy()
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


#endregion

#region NP

# Disable logging messages unless there is an error

set_log_level("ERROR")

confidence_level = 0.95

boundaries = round((1 - confidence_level) / 2, 2)
# NeuralProphet only accepts quantiles value in between 0 and 1
quantiles = [boundaries, confidence_level + boundaries]

# Model and prediction
m1 = NeuralProphet(
    quantiles=quantiles,
)
m1.set_plotting_backend("matplotlib")

m1.add_lagged_regressor('x')

m2 = NeuralProphet(
    quantiles=quantiles,
)
m2.set_plotting_backend("matplotlib")

m2.add_lagged_regressor('x')

m3 = NeuralProphet(
    quantiles=quantiles,
)
m3.set_plotting_backend("matplotlib")

m3.add_lagged_regressor('x')

m4 = NeuralProphet(
    quantiles=quantiles,
)
m4.set_plotting_backend("plotly")

m4.add_lagged_regressor('x')

m5 = NeuralProphet(
    quantiles=quantiles,
)
m5.set_plotting_backend("matplotlib")

m5.add_lagged_regressor('x')

metrics1 = m1.fit(df1_train, freq = 'D', epochs = 1000, validation_df = df1_test, progress = None)
metrics1

forecast1 = m1.predict(df1)
m1.plot(forecast1)

metrics2 = m2.fit(df2_train, freq = 'D', epochs = 1000, validation_df = df2_test, progress = None)
metrics2

forecast2 = m2.predict(df2)
m2.plot(forecast2)

metrics3 = m3.fit(df3_train, freq = 'D', epochs = 1000, validation_df = df3_test, progress = None)
metrics3

forecast3 = m3.predict(df3)
m3.plot(forecast3)

metrics4 = m4.fit(df4_train, freq = 'D', epochs = 1000, validation_df = df4_test, progress = None)
metrics4

forecast4 = m4.predict(df4)
m4.plot(forecast4)

metrics5 = m5.fit(df5_train, freq = 'D', epochs = 1000, validation_df = df5_test, progress = None)
metrics5

forecast5 = m5.predict(df5)
m5.plot(forecast5)

#endregion
