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

#clm = climatologies

clm_temp = pd.read_csv('/Users/rachelalaynahall/Desktop/NP/.conda/split_ct.csv')
clm_aod = pd.read_csv('/Users/rachelalaynahall/Desktop/NP/.conda/split_ca.csv')

#add name to date column

clm_temp.rename( columns={'Unnamed: 0':'Date'}, inplace=True )
clm_aod.rename( columns={'Unnamed: 0':'Date'}, inplace=True )


#convert to df

clm_temp = pd.DataFrame(clm_temp)
clm_aod = pd.DataFrame(clm_aod)


#remove ".X" from column headings

clm_temp.columns = clm_temp.columns.str.strip('X.')
clm_aod.columns = clm_aod.columns.str.strip('X.')

#convert date to datetime

clm_temp['Date'] = pd.to_datetime(clm_temp['Date'])
clm_aod['Date'] = pd.to_datetime(clm_aod['Date'])

#set date as index

#ct = climatologies temp
#ca = climatologies aod

ct = clm_temp.set_index('Date')
ca = clm_aod.set_index('Date')

#create features list
loc_list = list(ct)
features = loc_list

#create train and test variables of features

x_ct = ct.loc[:, features].values
x_ca = ca.loc[:, features].values

#create pca functions

pca = PCA(n_components = 5)


#fit pca functions on training data

pca.fit(x_ct)


#transform data through pca

principalComponents_ct = pca.transform(x_ct)
principalComponents_ca = pca.transform(x_ca)


#create new dfs with principal components

principalDf_ct = pd.DataFrame(data = principalComponents_ct
             , columns = ['principal component 1 - Temp', 'principal component 2 - Temp', 'principal component 3 - Temp', 'principal component 4 - Temp','principal component 5 - Temp'])
principalDf_ca = pd.DataFrame(data = principalComponents_ca
             , columns = ['principal component 1 - AOD', 'principal component 2 - AOD', 'principal component 3 - AOD', 'principal component 4 - AOD','principal component 5 - AOD'])


#add back in date column

principalDf_ct['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ct), freq='D')
principalDf_ca['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ca), freq='D')


#view explained variance

pca.explained_variance_ratio_

#view dfs

principalDf_ct
principalDf_ca

#separate train and test data

principalDf_ct['date'] = pd.to_datetime(principalDf_ct['date'])
principalDf_ca['date'] = pd.to_datetime(principalDf_ca['date'])

split_date = pd.datetime(1990,12,31)

ct_train = principalDf_ct.loc[principalDf_ct['date'] <= split_date]
ct_test = principalDf_ct.loc[principalDf_ct['date'] > split_date]

ca_train = principalDf_ca.loc[principalDf_ca['date'] <= split_date]
ca_test = principalDf_ca.loc[principalDf_ca['date'] > split_date]


#endregion

#region Create df for each PC

df1_test = ct_test['date'].copy()
temp1 = ct_test['principal component 1 - Temp'].copy()
aod1 = ca_test['principal component 1 - AOD'].copy()
df1_test = pd.concat([df1_test, temp1], axis = 1)
df1_test = pd.concat([df1_test, aod1], axis = 1)

df2_test = ct_test['date'].copy()
temp2 = ct_test[["principal component 2 - Temp"]].copy()
aod2 = ca_test[["principal component 2 - AOD"]].copy()
df2_test = pd.concat([df2_test, temp2], axis = 1)
df2_test = pd.concat([df2_test, aod2], axis = 1)

df3_test = ct_test['date'].copy()
temp3 = ct_test[["principal component 3 - Temp"]].copy()
aod3 = ca_test[["principal component 3 - AOD"]].copy()
df3_test = pd.concat([df3_test, temp3], axis = 1)
df3_test = pd.concat([df3_test, aod3], axis = 1)

df4_test = ct_test['date'].copy()
temp4 = ct_test[["principal component 4 - Temp"]] 
aod4 = ca_test[["principal component 4 - AOD"]].copy()
df4_test = pd.concat([df4_test, temp4], axis = 1)
df4_test = pd.concat([df4_test, aod4], axis = 1)

df5_test = ct_test['date'].copy()
temp5= ct_test[["principal component 5 - Temp"]].copy()
aod5 = ca_test[["principal component 5 - AOD"]].copy()
df5_test = pd.concat([df5_test, temp5], axis = 1)
df5_test = pd.concat([df5_test, aod5], axis = 1)

df1_train = ct_train['date'].copy()
temp1a = ct_train['principal component 1 - Temp'].copy()
aod1a = ca_train['principal component 1 - AOD'].copy()
df1_train = pd.concat([df1_train, temp1a], axis = 1)
df1_train = pd.concat([df1_train, aod1a], axis = 1)

df2_train = ct_train['date'].copy()
temp2a = ct_train[["principal component 2 - Temp"]].copy()
aod2a = ca_train[["principal component 2 - AOD"]].copy()
df2_train = pd.concat([df2_train, temp2a], axis = 1)
df2_train = pd.concat([df2_train, aod2a], axis = 1)

df3_train = ct_train['date'].copy()
temp3a = ct_train[["principal component 3 - Temp"]].copy()
aod3a = ca_train[["principal component 3 - AOD"]].copy()
df3_train = pd.concat([df3_train, temp3a], axis = 1)
df3_train = pd.concat([df3_train, aod3a], axis = 1)

df4_train = ct_train['date'].copy()
temp4a = ct_train[["principal component 4 - Temp"]].copy()
aod4a = ca_train[["principal component 4 - AOD"]].copy()
df4_train = pd.concat([df4_train, temp4a], axis = 1)
df4_train = pd.concat([df4_train, aod4a], axis = 1)

df5_train = ct_train['date'].copy()
temp5a = ct_train[["principal component 5 - Temp"]].copy()
aod5a = ca_train[["principal component 5 - AOD"]].copy()
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

# region Neural Prophet

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
