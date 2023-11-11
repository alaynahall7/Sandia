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

pca_ct = PCA(n_components = 10)
pca_ca = PCA(n_components = 10)

#fit pca functions on training data

pca_ct.fit(x_ct)
pca_ca.fit(x_ca)

#transform data through pca

principalComponents_ct = pca_ct.transform(x_ct)
principalComponents_ca = pca_ca.transform(x_ca)


#create new dfs with principal components

principalDf_ct = pd.DataFrame(data = principalComponents_ct
             , columns = ['principal component 1 - Temp', 'principal component 2 - Temp', 'principal component 3 - Temp', 'principal component 4 - Temp','principal component 5 - Temp', 'principal component 6 - Temp', 'principal component 7 - Temp', 'principal component 8 - Temp', 'principal component 9 - Temp','principal component 10 - Temp'])

principalDf_ca = pd.DataFrame(data = principalComponents_ca
             , columns = ['principal component 1 - AOD', 'principal component 2 - AOD', 'principal component 3 - AOD', 'principal component 4 - AOD','principal component 5 - AOD', 'principal component 6 - AOD', 'principal component 7 - AOD', 'principal component 8 - AOD', 'principal component 9 - AOD','principal component 10 - AOD'])


#add back in date column
principalDf_ct['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ct), freq='D')
principalDf_ca['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ca), freq='D')


#view explained variance
pca_ct.explained_variance_ratio_
pca_ca.explained_variance_ratio_

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


#region Create dataframe for each principal component

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

df6_test = ct_test['date'].copy()
temp6 = ct_test[["principal component 6 - Temp"]].copy()
aod6 = ca_test[["principal component 6 - AOD"]].copy()
df6_test = pd.concat([df6_test, temp6], axis = 1)
df6_test = pd.concat([df6_test, aod6], axis = 1)

df7_test = ct_test['date'].copy()
temp7 = ct_test[["principal component 7 - Temp"]].copy()
aod7 = ca_test[["principal component 7 - AOD"]].copy()
df7_test = pd.concat([df7_test, temp7], axis = 1)
df7_test = pd.concat([df7_test, aod7], axis = 1)

df8_test = ct_test['date'].copy()
temp8 = ct_test[["principal component 8 - Temp"]].copy()
aod8 = ca_test[["principal component 8 - AOD"]].copy()
df8_test = pd.concat([df8_test, temp8], axis = 1)
df8_test = pd.concat([df8_test, aod8], axis = 1)

df9_test = ct_test['date'].copy()
temp9 = ct_test[["principal component 9 - Temp"]].copy()
aod9 = ca_test[["principal component 9 - AOD"]].copy()
df9_test = pd.concat([df9_test, temp9], axis = 1)
df9_test = pd.concat([df9_test, aod9], axis = 1)

df10_test = ct_test['date'].copy()
temp10 = ct_test[["principal component 10 - Temp"]].copy()
aod10 = ca_test[["principal component 10 - AOD"]].copy()
df10_test = pd.concat([df10_test, temp10], axis = 1)
df10_test = pd.concat([df10_test, aod10], axis = 1)

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

df6_train = ct_train['date'].copy()
temp6a = ct_train[["principal component 6 - Temp"]].copy()
aod6a = ca_train[["principal component 6 - AOD"]].copy()
df6_train = pd.concat([df6_train, temp6a], axis = 1)
df6_train = pd.concat([df6_train, aod6a], axis = 1)

df7_train = ct_train['date'].copy()
temp7a = ct_train[["principal component 7 - Temp"]].copy()
aod7a = ca_train[["principal component 7 - AOD"]].copy()
df7_train = pd.concat([df7_train, temp7a], axis = 1)
df7_train = pd.concat([df7_train, aod7a], axis = 1)

df8_train = ct_train['date'].copy()
temp8a = ct_train[["principal component 8 - Temp"]].copy()
aod8a = ca_train[["principal component 8 - AOD"]].copy()
df8_train = pd.concat([df8_train, temp8a], axis = 1)
df8_train = pd.concat([df8_train, aod8a], axis = 1)

df9_train = ct_train['date'].copy()
temp9a = ct_train[["principal component 9 - Temp"]].copy()
aod9a = ca_train[["principal component 9 - AOD"]].copy()
df9_train = pd.concat([df9_train, temp9a], axis = 1)
df9_train = pd.concat([df9_train, aod9a], axis = 1)

df10_train = ct_train['date'].copy()
temp10a = ct_train[["principal component 10 - Temp"]].copy()
aod10a = ca_train[["principal component 10 - AOD"]].copy()
df10_train = pd.concat([df10_train, temp10a], axis = 1)
df10_train = pd.concat([df10_train, aod10a], axis = 1)

df1_train.rename(columns = {'date':'ds'}, inplace = True)
df2_train.rename(columns = {'date':'ds'}, inplace = True)
df3_train.rename(columns = {'date':'ds'}, inplace = True)
df4_train.rename(columns = {'date':'ds'}, inplace = True)
df5_train.rename(columns = {'date':'ds'}, inplace = True)
df6_train.rename(columns = {'date':'ds'}, inplace = True)
df7_train.rename(columns = {'date':'ds'}, inplace = True)
df8_train.rename(columns = {'date':'ds'}, inplace = True)
df9_train.rename(columns = {'date':'ds'}, inplace = True)
df10_train.rename(columns = {'date':'ds'}, inplace = True)

df1_test.rename(columns = {'date':'ds'}, inplace = True)
df2_test.rename(columns = {'date':'ds'}, inplace = True)
df3_test.rename(columns = {'date':'ds'}, inplace = True)
df4_test.rename(columns = {'date':'ds'}, inplace = True)
df5_test.rename(columns = {'date':'ds'}, inplace = True)
df6_test.rename(columns = {'date':'ds'}, inplace = True)
df7_test.rename(columns = {'date':'ds'}, inplace = True)
df8_test.rename(columns = {'date':'ds'}, inplace = True)
df9_test.rename(columns = {'date':'ds'}, inplace = True)
df10_test.rename(columns = {'date':'ds'}, inplace = True)

df1_train.rename(columns = {'principal component 1 - Temp':'y'}, inplace = True)
df2_train.rename(columns = {'principal component 2 - Temp':'y'}, inplace = True)
df3_train.rename(columns = {'principal component 3 - Temp':'y'}, inplace = True)
df4_train.rename(columns = {'principal component 4 - Temp':'y'}, inplace = True)
df5_train.rename(columns = {'principal component 5 - Temp':'y'}, inplace = True)
df6_train.rename(columns = {'principal component 6 - Temp':'y'}, inplace = True)
df7_train.rename(columns = {'principal component 7 - Temp':'y'}, inplace = True)
df8_train.rename(columns = {'principal component 8 - Temp':'y'}, inplace = True)
df9_train.rename(columns = {'principal component 9 - Temp':'y'}, inplace = True)
df10_train.rename(columns = {'principal component 10 - Temp':'y'}, inplace = True)

df1_test.rename(columns = {'principal component 1 - Temp':'y'}, inplace = True)
df2_test.rename(columns = {'principal component 2 - Temp':'y'}, inplace = True)
df3_test.rename(columns = {'principal component 3 - Temp':'y'}, inplace = True)
df4_test.rename(columns = {'principal component 4 - Temp':'y'}, inplace = True)
df5_test.rename(columns = {'principal component 5 - Temp':'y'}, inplace = True)
df6_test.rename(columns = {'principal component 6 - Temp':'y'}, inplace = True)
df7_test.rename(columns = {'principal component 7 - Temp':'y'}, inplace = True)
df8_test.rename(columns = {'principal component 8 - Temp':'y'}, inplace = True)
df9_test.rename(columns = {'principal component 9 - Temp':'y'}, inplace = True)
df10_test.rename(columns = {'principal component 10 - Temp':'y'}, inplace = True)

df1_train.rename(columns = {'principal component 1 - AOD':'x'}, inplace = True)
df2_train.rename(columns = {'principal component 2 - AOD':'x'}, inplace = True)
df3_train.rename(columns = {'principal component 3 - AOD':'x'}, inplace = True)
df4_train.rename(columns = {'principal component 4 - AOD':'x'}, inplace = True)
df5_train.rename(columns = {'principal component 5 - AOD':'x'}, inplace = True)
df6_train.rename(columns = {'principal component 6 - AOD':'x'}, inplace = True)
df7_train.rename(columns = {'principal component 7 - AOD':'x'}, inplace = True)
df8_train.rename(columns = {'principal component 8 - AOD':'x'}, inplace = True)
df9_train.rename(columns = {'principal component 9 - AOD':'x'}, inplace = True)
df10_train.rename(columns = {'principal component 10 - AOD':'x'}, inplace = True)


df1_test.rename(columns = {'principal component 1 - AOD':'x'}, inplace = True)
df2_test.rename(columns = {'principal component 2 - AOD':'x'}, inplace = True)
df3_test.rename(columns = {'principal component 3 - AOD':'x'}, inplace = True)
df4_test.rename(columns = {'principal component 4 - AOD':'x'}, inplace = True)
df5_test.rename(columns = {'principal component 5 - AOD':'x'}, inplace = True)
df6_test.rename(columns = {'principal component 6 - AOD':'x'}, inplace = True)
df7_test.rename(columns = {'principal component 7 - AOD':'x'}, inplace = True)
df8_test.rename(columns = {'principal component 8 - AOD':'x'}, inplace = True)
df9_test.rename(columns = {'principal component 9 - AOD':'x'}, inplace = True)
df10_test.rename(columns = {'principal component 10 - AOD':'x'}, inplace = True)


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

df6 = df6_train
df6 = pd.concat([df6, df6_test], axis = 0)

df7 = df7_train
df7 = pd.concat([df7, df7_test], axis = 0)

df8 = df8_train
df8 = pd.concat([df8, df8_test], axis = 0)

df9 = df9_train
df9 = pd.concat([df9, df9_test], axis = 0)

df10 = df10_train
df10 = pd.concat([df10, df10_test], axis = 0)





#endregion

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

m6 = NeuralProphet(
    quantiles=quantiles,
)
m6.set_plotting_backend("matplotlib")

m6.add_lagged_regressor('x')

m7 = NeuralProphet(
    quantiles=quantiles,
)
m7.set_plotting_backend("matplotlib")

m7.add_lagged_regressor('x')

m8 = NeuralProphet(
    quantiles=quantiles,
)
m8.set_plotting_backend("matplotlib")

m8.add_lagged_regressor('x')

m9 = NeuralProphet(
    quantiles=quantiles,
)
m9.set_plotting_backend("matplotlib")

m9.add_lagged_regressor('x')

m10 = NeuralProphet(
    quantiles=quantiles,
)
m10.set_plotting_backend("matplotlib")

m10.add_lagged_regressor('x')

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

metrics6 = m6.fit(df6_train, freq = 'D', epochs = 1000, validation_df = df6_test, progress = None)
metrics6

forecast6 = m6.predict(df6)
m6.plot(forecast6)

metrics7 = m7.fit(df7_train, freq = 'D', epochs = 1000, validation_df = df7_test, progress = None)
metrics7

forecast7 = m7.predict(df7)
m7.plot(forecast7)

metrics8 = m8.fit(df8_train, freq = 'D', epochs = 1000, validation_df = df8_test, progress = None)
metrics8

forecast8 = m8.predict(df8)
m8.plot(forecast8)

metrics9 = m9.fit(df9_train, freq = 'D', epochs = 1000, validation_df = df9_test, progress = None)
metrics9

forecast9 = m9.predict(df9)
m9.plot(forecast9)

metrics10 = m10.fit(df10_train, freq = 'D', epochs = 1000, validation_df = df10_test, progress = None)
metrics10

forecast10 = m10.predict(df10)
m10.plot(forecast10)
