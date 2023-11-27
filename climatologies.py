import numpy as np
import pandas as pd

import xarray as xr
import hvplot.xarray
import hvplot.pandas

import matplotlib.pyplot as plt

import matplotlib.path as mpath
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from neuralprophet import NeuralProphet, set_log_level

# CDS API
import cdsapi

# Libraries for plotting and visualising data
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

# Disable warnings for data download via API
import urllib3 
urllib3.disable_warnings()

from pathlib import Path  

from matplotlib.animation import FuncAnimation
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature
from scipy.integrate import ode
from IPython.display import HTML
from tempfile import NamedTemporaryFile
from matplotlib import animation





#region Data setup

#import the data

#clm = climatologies

clm_temp = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/tavg3_3d_asm_Nv_daily_48x24.csv')
clm_aod = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/TOTEXTTAU_daily_48x24.csv')

clm_temp['date'] = pd.to_datetime(clm_temp['date'])
clm_aod['date'] = pd.to_datetime(clm_aod['date'])

clm_temp.rename(columns = {'date':'time'}, inplace = True)
clm_aod.rename(columns = {'date':'time'}, inplace = True)
clm_aod.rename(columns = {'TOTEXTTAU':'A'}, inplace = True)

enddate = pd.datetime(1998,12,10)
clm_aod = clm_aod[clm_aod['time'] <= enddate]

dt = clm_temp.set_index(["time", "lat", "lon"]).to_xarray()
ct_period = dt.sel(time = slice('1986-01-01', '1990-12-31'))
ct_month = ct_period.groupby('time.month').mean()
at_month = dt.groupby('time.month') - ct_month

da = clm_aod.set_index(["time", "lat", "lon"]).to_xarray()
ca_period = da.sel(time = slice('1986-01-01', '1990-12-31'))
ca_month = ca_period.groupby('time.month').mean()
aa_month = da.groupby('time.month') - ca_month

ct = at_month.to_dataframe()
ca = aa_month.to_dataframe()


filepath = Path('/Users/rachelalaynahall/Desktop/ct.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
ct.to_csv(filepath)

 
filepath2 = Path('/Users/rachelalaynahall/Desktop/ca.csv')  
filepath2.parent.mkdir(parents=True, exist_ok=True)  
ca.to_csv(filepath2)

ct = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/ct.csv')
ca = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/ca.csv')

ca.rename(columns = {'TOTEXTTAU':'A'}, inplace = True)

ct["lat"] = ct["lat"] + 90
ct["lon"] = ct["lon"] + 180

ca["lat"] = ca["lat"] + 90
ca["lon"] = ca["lon"] + 180

ct["location"] = ct["lat"].astype(str) + ".." + ct["lon"].astype(str)

ca["location"] = ca["lat"].astype(str) + ".." + ca["lon"].astype(str)


ct = ct.sort_values(by = ['location', 'time'], ascending = [True, True])
ca = ca.sort_values(by = ['location', 'time'], ascending = [True, True])

ct = ct.set_index('time')
ca = ca.set_index('time')

ct = ct.drop(['lat', 'lon', 'month'], axis = 1)
ca = ca.drop(['lat', 'lon', 'month'], axis = 1)

  
filepath3 = Path('/Users/rachelalaynahall/Desktop/ct1.csv')  
filepath3.parent.mkdir(parents=True, exist_ok=True)  
ct.to_csv(filepath3)


filepath4 = Path('/Users/rachelalaynahall/Desktop/ca1.csv')  
filepath4.parent.mkdir(parents=True, exist_ok=True)  
ca.to_csv(filepath4)

#endregion

#region PCA

#import the data

#clm = climatologies

clm_t = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/splitct.csv')
clm_a = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/splitca.csv')

#add name to date column

clm_t.rename( columns={'Unnamed: 0':'Date'}, inplace=True )
clm_a.rename( columns={'Unnamed: 0':'Date'}, inplace=True )


#convert to df

clm_t = pd.DataFrame(clm_t)
clm_a = pd.DataFrame(clm_a)


#remove ".X" from column headings

clm_t.columns = clm_t.columns.str.strip('X.')
clm_a.columns = clm_a.columns.str.strip('X.')

#convert date to datetime

clm_t['Date'] = pd.to_datetime(clm_t['Date'])
clm_a['Date'] = pd.to_datetime(clm_a['Date'])

#set date as index

#ct = climatologies temp
#ca = climatologies aod

clm_t = clm_t.set_index('Date')
clm_a = clm_a.set_index('Date')

#create features list
loc_list = list(clm_t)
features = loc_list

#create train and test variables of features

x_ct = clm_t.loc[:, features].values
x_ca = clm_a.loc[:, features].values

#create scaler functions

scaler_ct = StandardScaler()
scaler_ca = StandardScaler()

#fit scalers on training data

scaler_ct.fit(x_ct)
scaler_ca.fit(x_ca)

#scale data

x_ct = scaler_ct.transform(x_ct)
x_ca = scaler_ca.transform(x_ca)

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
    growth = 'discontinuous',
    n_lags= 30,
    seasonality_mode= 'multiplicative',
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m1.set_plotting_backend("matplotlib")

m1.add_lagged_regressor('x')

m2 = NeuralProphet(
    growth = 'discontinuous',
    n_lags= 30,
    seasonality_mode= 'multiplicative',
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m2.set_plotting_backend("matplotlib")

m2.add_lagged_regressor('x')

m3 = NeuralProphet(
    growth = 'discontinuous',
    n_lags= 30,
    seasonality_mode= 'multiplicative',
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m3.set_plotting_backend("matplotlib")

m3.add_lagged_regressor('x')

m4 = NeuralProphet(
    growth = 'discontinuous',
    n_lags= 30,
    seasonality_mode= 'multiplicative',
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m4.set_plotting_backend("matplotlib")

m4.add_lagged_regressor('x')

m5 = NeuralProphet(
    growth = 'discontinuous',
    n_lags= 30,
    seasonality_mode= 'multiplicative',
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m5.set_plotting_backend("matplotlib")

m5.add_lagged_regressor('x')

metrics1 = m1.fit(df1_train, freq = 'D', epochs = 400, validation_df = df1_test, progress = None)
metrics1

forecast1 = m1.predict(df1)
m1.plot(forecast1)

metrics2 = m2.fit(df2_train, freq = 'D', epochs = 400, validation_df = df2_test, progress = None)
metrics2

forecast2 = m2.predict(df2)
m2.plot(forecast2)

metrics3 = m3.fit(df3_train, freq = 'D', epochs = 400, validation_df = df3_test, progress = None)
metrics3

forecast3 = m3.predict(df3)
m3.plot(forecast3)

metrics4 = m4.fit(df4_train, freq = 'D', epochs = 400, validation_df = df4_test, progress = None)
metrics4

forecast4 = m4.predict(df4)
m4.plot(forecast4)

metrics5 = m5.fit(df5_train, freq = 'D', epochs = 400, validation_df = df5_test, progress = None)
metrics5

forecast5 = m5.predict(df5)
m5.plot(forecast5)



fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics1["MAE"], '-o', label="Training Loss")  
ax.plot(metrics1["MAE_val"], '-r', label="Validation Loss")
ax.legend(loc='center right', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics2["MAE"], '-o', label="Training Loss")  
ax.plot(metrics2["MAE_val"], '-r', label="Validation Loss")
ax.legend(loc='center right', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics3["MAE"], '-o', label="Training Loss")  
ax.plot(metrics3["MAE_val"], '-r', label="Validation Loss")
ax.legend(loc='center right', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics4["MAE"], '-o', label="Training Loss")  
ax.plot(metrics4["MAE_val"], '-r', label="Validation Loss")
ax.legend(loc='center right', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics5["MAE"], '-o', label="Training Loss")  
ax.plot(metrics5["MAE_val"], '-r', label="Validation Loss")
ax.legend(loc='center right', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)



#endregion

# region Backscale


yhat1 = forecast1.loc[forecast1['ds'] > split_date]
df1_predict = df1_train['y'].copy()
yhat1 = yhat1['yhat1'].copy()
df1_predict = pd.concat([df1_predict, yhat1], axis = 0)

yhat2 = forecast2.loc[forecast2['ds'] > split_date]
df2_predict = df2_train['y'].copy()
yhat2 = yhat2['yhat1'].copy()
df2_predict = pd.concat([df2_predict, yhat2], axis = 0)

yhat3 = forecast3.loc[forecast3['ds'] > split_date]
df3_predict = df3_train['y'].copy()
yhat3 = yhat3['yhat1'].copy()
df3_predict = pd.concat([df3_predict, yhat3], axis = 0) 

yhat4 = forecast4.loc[forecast4['ds'] > split_date]
df4_predict = df4_train['y'].copy()
yhat4 = yhat4['yhat1'].copy()
df4_predict = pd.concat([df4_predict, yhat4], axis = 0)

yhat5 = forecast5.loc[forecast5['ds'] > split_date]
df5_predict = df5_train['y'].copy()
yhat5 = yhat5['yhat1'].copy()
df5_predict = pd.concat([df5_predict, yhat5], axis = 0)

df_predict = pd.concat([df1_predict, df2_predict, df3_predict, df4_predict, df5_predict], axis = 1)


ct_orig = np.dot(df_predict,pca.components_)
ct_orig_backscaled = scaler_ct.inverse_transform(ct_orig)

ct_orig = pd.DataFrame(ct_orig)
ct_orig_backscaled = pd.DataFrame(ct_orig_backscaled)

ct_orig_backscaled['date'] = pd.date_range(start='1/1/1986', periods = len(principalDf_ct), freq='D')

ct_orig_backscaled = ct_orig_backscaled.set_index('date')
ct_orig_backscaled.columns = clm_t.columns

 
filepath5 = Path('/Users/rachelalaynahall/Desktop/ct_bs.csv')  
filepath5.parent.mkdir(parents=True, exist_ok=True)  
ct_orig_backscaled.to_csv(filepath5)

ct_bs = pd.read_csv('/Users/rachelalaynahall/Desktop/Desktop - Joshua’s MacBook Air/NP/.conda/ct_bs.csv')

#ct_bs[['lat', 'lon']] = ct_bs.location.str.split("..", expand = True)
ct_bs[['lat', 'lon']] = ct_bs["location"].apply(lambda x: pd.Series(str(x).split("..")))

ct_bs['lat'] = ct_bs['lat'].str.replace('X', '')
ct_bs["lat"] = ct_bs["lat"].astype(float)
ct_bs["lon"] = ct_bs["lon"].astype(float)
ct_bs["lat"] = ct_bs["lat"] - 90
ct_bs["lon"] = ct_bs["lon"] - 180
ct_bs = ct_bs.drop(['location'], axis = 1)
ct_bs['date'] = pd.to_datetime(ct_bs['date'])
ct_bs.rename(columns = {'date':'time'}, inplace = True)

ct_bs_array = ct_bs.set_index(["time", "lat", "lon"]).to_xarray()
dtb_month = ct_bs_array.groupby('time.month') + ct_month

tf = dtb_month.to_dataframe()
tf = tf.drop(['month'], axis = 1)
compare = dt.to_dataframe()
 
filepath6 = Path('/Users/rachelalaynahall/Desktop/tf.csv')  
filepath6.parent.mkdir(parents=True, exist_ok=True)  
tf.to_csv(filepath6)

filepath7 = Path('/Users/rachelalaynahall/Desktop/compare.csv')  
filepath7.parent.mkdir(parents=True, exist_ok=True)  
compare.to_csv(filepath7)


#endregion


#region Plots

dtb_month.hvplot(groupby = 'time')
ct_bs_array.hvplot(groupby = 'time')


# First we specify Coordinate Refference System for Map Projection
# We will use Mercator, which is a cylindrical, conformal projection. 
# It has bery large distortion at high latitudes, cannot 
# fully reach the polar regions.
projection = ccrs.Mercator()

# Specify CRS, that will be used to tell the code, where should our data be plotted
crs = ccrs.PlateCarree()

# Now we will create axes object having specific projection 
plt.figure(dpi=150)
ax = plt.axes(projection=projection, frameon=True)

# Draw gridlines in degrees over Mercator map
gl = ax.gridlines(crs=crs, draw_labels=True,
                  linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
gl.xlabel_style = {"size" : 7}
gl.ylabel_style = {"size" : 7}

# To plot borders and coastlines, we can use cartopy feature

ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

# Now, we will specify extent of our map in minimum/maximum longitude/latitude
# Note that these values are specified in degrees of longitude and degrees of latitude
# However, we can specify them in any crs that we want, but we need to provide appropriate
# crs argument in ax.set_extent
lon_min = -176.6
lon_max = 175.9
lat_min = -86.0
lat_max = 86.5

# crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
#plt.title(f"Temperature anomaly over Europe in {original_data.valid_time.dt.strftime('%B %Y').values}")
plt.show()



#Creating an animation for 1950-2020 and saving it as an MP4 video.
cbar_kwargs = {
    'orientation':'horizontal',
    'fraction': 0.048,
    'pad': 0.01,
    'extend':'neither'
}

fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(left=0.02, bottom=0.04, right=0.98, top=0.96)
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())
ax.add_feature(NaturalEarthFeature('cultural', 'admin_0_countries', '10m'),
              facecolor='none', edgecolor='black')
ax.set_extent([-176.6, 175.9, -86.0, 86.5])

image = dtb_month.isel(time=0).plot.imshow(ax=ax, add_labels=False,
                       vmin=-4, vmax=4, cmap='coolwarm', animated=True,
                       cbar_kwargs=cbar_kwargs, interpolation='bicubic')



def animate(t):
    date =  pd.to_datetime(dtb_month.sel(time=t)['time'].values)
    ax.set_title("Temperature in " + str(date.year))
    ax.title.set_fontsize(18)
    image.set_array(dtb_month.sel(time=t))
    return image

ani = FuncAnimation(fig, animate, frames=dtb_month['time'].values[-71:], blit=False)
ani.save("animation.mp4", fps=2, extra_args=['-vcodec','libx264', '-crf','15', '-preset','veryslow'])

#endregion
