#!/usr/bin/env python
# coding: utf-8

# # Master Thesis: Drought prediction using Data Science and Machine Learning methods - a case study from Botswana

# ### by Hannah Kemper, Geographic Institute University Bonn 
# 
# Summer term 2021, contact: hannahkemper.speyer@gmail.com
# 
# #### Supervision by Prof. Dr. Klaus Greve (University Bonn) & Dr. Felicia Akinyemi (University Bern)

# #### Research on the following question: 
# ##### Which natural factors indicating drought periods impact the agricultural productivity of rainfed crop production in Botswana and are significant considering the implementation of a Drought Early Warning System?
# 
# _Four sub questions are analyzed in the following script (more details in the according section)_
# 
#  __Research target:__ find reliable information about drought dynamics in Botswana and ways to predict losses in agricultural productivity using drought and climate indicators by applying a data science and machine learning workflow
#  
#  
#  

# >**Importing python libraries needed for the analysis**

# In[1]:


import pandas as pd

import numpy as np
from numpy import where
import pysal as ps

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import scipy as scipy
import scipy.stats as stats
from scipy.stats import skew

from math import sqrt

import statsmodels.api as sm
from statsmodels.formula.api import ols

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

from collections import Counter

import changefinder
import ruptures as rpt

# Import from Scikit Learn library

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.datasets import make_classification

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error


# >**Setting font size and color schemes of matplotlib plot function**

# In[2]:


plt.rcParams.update({'font.size':15})
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=15)


# ***

# ***

# ### 0. Data Import
# 
# The Data Import is done using pandas read function using a csv of the complete dataset

# In[3]:


# We will use the complete dataset that contains all variables and a boolean identifier of emDat drought periods


df = pd.read_csv('DATABASE_restart.csv', encoding= 'ISO-8859-1',
                 delimiter=';', decimal = '.')


# In[4]:


# Creating a copy of original pandas dataframe
df_total = df


# In[5]:


# We also use the national dataset in SQ4, so we import a csv with the needed information

df_national = pd.read_csv('DATABASE_national_allvariables.csv', encoding= 'ISO-8859-1',
                 delimiter=';', decimal = '.')


# ### 1. Data Wrangling
# 
# Data Wrangling is needed to detect missing values and outliers in the dataset that could affect the reliability and plausibility of correlations and relationships between the variables
# 
# We will first describe the data set to gain an impression on it

# In[6]:


pd.set_option('max_columns', None)
df_total.describe()


# In[7]:


# How many values are missing of each variable?

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
df_total.isnull().sum()


# #### We need to handle the missing values

# In[8]:


# Imputing mean values for productivity, removing outliers of 1.5 interquartile range

df_total['productivity'].fillna((df_total['productivity'].mean()), inplace=True)

q_low = df_total['productivity'].quantile(0.01)
q_high  = df_total['productivity'].quantile(0.99)

iqr = q_high-q_low #Interquartile range
fence_low  = q_low-1.5*iqr
fence_high = q_high+1.5*iqr

df_out = df_total.loc[(df_total['productivity'] > fence_low) & (df_total['productivity'] < fence_high)]

df_total=df_out


# In[9]:


# Imputing mean values for yield_kgha, removing outliers of 1.5 interquartile range

df_total['yield_kgha'].fillna((df_total['yield_kgha'].mean()), inplace=True)

q_low = df_total['yield_kgha'].quantile(0.01)
q_high  = df_total['yield_kgha'].quantile(0.99)

iqr = q_high-q_low #Interquartile range
fence_low  = q_low-1.5*iqr
fence_high = q_high+1.5*iqr

df_out = df_total.loc[(df_total['yield_kgha'] > fence_low) & (df_total['yield_kgha'] < fence_high)]

df_total=df_out


# In[10]:


# How many values are missing in percent?

for col in df_total.columns:
    pct_missing = np.mean(df_total[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[11]:


# For the columns with values missing, we impute NAs with mean values

df_clean = df_total

df_clean['EVI_3'].fillna((df_total['EVI_3'].mean()), inplace=True)
df_clean['EVI_12'].fillna((df_total['EVI_12'].mean()), inplace=True)
df_clean['NDVI_3'].fillna((df_total['NDVI_3'].mean()), inplace=True)
df_clean['NDVI_12'].fillna((df_total['NDVI_12'].mean()), inplace=True)

df_clean['NDWI_3'].fillna((df_total['NDWI_3'].mean()), inplace=True)
df_clean['NDWI_12'].fillna((df_total['NDWI_12'].mean()), inplace=True)
df_clean['PRECIPITATION_3'].fillna((df_total['PRECIPITATION_3'].mean()), inplace=True)
df_clean['PRECIPITATION_12'].fillna((df_total['PRECIPITATION_12'].mean()), inplace=True)

df_clean['TMIN_12'].fillna((df_total['TMIN_12'].mean()), inplace=True)
df_clean['TMIN_3'].fillna((df_total['TMIN_3'].mean()), inplace=True)
df_clean['TMAX_12'].fillna((df_total['TMAX_12'].mean()), inplace=True)
df_clean['TMAX_3'].fillna((df_total['TMAX_3'].mean()), inplace=True)

df_clean['SOILMOISTURE_3'].fillna((df_total['SOILMOISTURE_3'].mean()), inplace=True)
df_clean['SOILMOISTURE_12'].fillna((df_total['SOILMOISTURE_12'].mean()), inplace=True)

df_clean['WINDSPEED_3'].fillna((df_total['WINDSPEED_3'].mean()), inplace=True)
df_clean['PDSI_3'].fillna((df_total['PDSI_3'].mean()), inplace=True)
df_clean['PDSI_12'].fillna((df_total['PDSI_12'].mean()), inplace=True)

df_clean['VCI_3'].fillna((df_total['VCI_3'].mean()), inplace=True)
df_clean['VCI_12'].fillna((df_total['VCI_12'].mean()), inplace=True)

df_clean['VHI_3'].fillna((df_total['VHI_3'].mean()), inplace=True)
df_clean['VHI_12'].fillna((df_total['VHI_12'].mean()), inplace=True)


# In[12]:


for col in df_clean.columns:
    pct_missing = np.mean(df_clean[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[13]:


# How is the shape of the cleaned dataset?
df_clean.shape


# ### Standardization of Variables
# 
# Standardization of the Variables is needed to conduct reliable correlation analysis
# it is important to scale the variables on a scale from -1 to 1 with a mean value of 0 to have a Gaussian distribution
# 
# a new dataframe called "cdf_standard" is created with standardized values for each numeric column

# In[14]:


scaler = StandardScaler()

# We make a copy of all float type columns of our dataset and call it standard, 
# so we can apply the standardscaler to the columns

df_standard = df_clean[['yield_kgha','import','NAOI_3','NAOI_12', 'SOI_3', 
             'SOI_12','productivity', 'EVI_3', 'EVI_12', 'NDVI_3',
             'NDVI_12', 'NDWI_3', 'NDWI_12', 'PRECIPITATION_3',
             'PRECIPITATION_12', 'TMIN_3', 'TMIN_12', 'TAVG_3',
             'TAVG_12', 'TMAX_3', 'TMAX_12', 'SOILMOISTURE_3',
             'SOILMOISTURE_12', 'WINDSPEED_3', 'WINDSPEED_12',
             'PDSI_3', 'PDSI_12', 'VCI_3', 'VCI_12', 'TCI_3', 'TCI_12',
             'VHI_3', 'VHI_12', 'SPI_3', 'SPI_12']].copy()


# In[15]:


#We apply the standardscaler

df_standard[['yield_kgha','import','NAOI_3','NAOI_12', 'SOI_3', 
             'SOI_12','productivity', 'EVI_3', 'EVI_12', 'NDVI_3',
             'NDVI_12', 'NDWI_3', 'NDWI_12', 'PRECIPITATION_3',
             'PRECIPITATION_12', 'TMIN_3', 'TMIN_12', 'TAVG_3',
             'TAVG_12', 'TMAX_3', 'TMAX_12', 'SOILMOISTURE_3',
             'SOILMOISTURE_12', 'WINDSPEED_3', 'WINDSPEED_12',
             'PDSI_3', 'PDSI_12', 'VCI_3', 'VCI_12', 'TCI_3', 'TCI_12',
             'VHI_3', 'VHI_12', 'SPI_3', 'SPI_12'
            ]] = scaler.fit_transform(df_standard[['yield_kgha','import','NAOI_3','NAOI_12', 'SOI_3', 
             'SOI_12','productivity', 'EVI_3', 'EVI_12', 'NDVI_3',
             'NDVI_12', 'NDWI_3', 'NDWI_12', 'PRECIPITATION_3',
             'PRECIPITATION_12', 'TMIN_3', 'TMIN_12', 'TAVG_3',
             'TAVG_12', 'TMAX_3', 'TMAX_12', 'SOILMOISTURE_3',
             'SOILMOISTURE_12', 'WINDSPEED_3', 'WINDSPEED_12',
             'PDSI_3', 'PDSI_12', 'VCI_3', 'VCI_12', 'TCI_3', 'TCI_12',
             'VHI_3', 'VHI_12', 'SPI_3', 'SPI_12']])


# In[16]:


#Check whether there are any string type columns in our dataset
df_standard.info()


# In[17]:


# We use the numpy function where to replace values between 0 and 0.001 with 0 to get rid of infinity values

df_standard['yield_kgha'].where(~(df_standard.yield_kgha > 0) & (df_standard.yield_kgha < 0.0001), other = 0, inplace =True)
#df_standard['import'].where(~(df_standard.import > 0) & (df_standard.import < 0.0001), other = 0, inplace =True)
df_standard['productivity'].where(~(df_standard.productivity > 0) & (df_standard.productivity < 0.0001), other = 0, inplace =True)

df_standard['NAOI_3'].where(~(df_standard.NAOI_3 > 0) & (df_standard.NAOI_3 < 0.0001), other = 0, inplace =True)
df_standard['NAOI_12'].where(~(df_standard.NAOI_12 > 0) & (df_standard.NAOI_12 < 0.0001), other = 0, inplace =True)
df_standard['SOI_3'].where(~(df_standard.SOI_3 > 0) & (df_standard.SOI_3 < 0.0001), other = 0, inplace =True)
df_standard['SOI_12'].where(~(df_standard.SOI_12 > 0) & (df_standard.SOI_12 < 0.0001), other = 0, inplace =True)

df_standard['EVI_3'].where(~(df_standard.EVI_3 > 0) & (df_standard.EVI_3 < 0.0001), other = 0, inplace =True)
df_standard['EVI_12'].where(~(df_standard.EVI_12 > 0) & (df_standard.EVI_12 < 0.0001), other = 0, inplace =True)
df_standard['NDVI_3'].where(~(df_standard.NDVI_3 > 0) & (df_standard.NDVI_3 < 0.0001), other = 0, inplace =True)
df_standard['NDVI_12'].where(~(df_standard.NDVI_12 > 0) & (df_standard.NDVI_12 < 0.0001), other = 0, inplace =True)

df_standard['NDWI_3'].where(~(df_standard.NDWI_3 > 0) & (df_standard.NDWI_3 < 0.0001), other = 0, inplace =True)
df_standard['NDWI_12'].where(~(df_standard.NDWI_12 > 0) & (df_standard.NDWI_12 < 0.0001), other = 0, inplace =True)

df_standard['PRECIPITATION_3'].where(~(df_standard.PRECIPITATION_3 > 0) & (df_standard.PRECIPITATION_3 < 0.0001), other = 0, inplace =True)
df_standard['PRECIPITATION_12'].where(~(df_standard.PRECIPITATION_12 > 0) & (df_standard.PRECIPITATION_12 < 0.0001), other = 0, inplace =True)
df_standard['TMIN_3'].where(~(df_standard.TMIN_3 > 0) & (df_standard.TMIN_3 < 0.0001), other = 0, inplace =True)
df_standard['TMIN_12'].where(~(df_standard.TMIN_12 > 0) & (df_standard.TMIN_12 < 0.0001), other = 0, inplace =True)
df_standard['TAVG_3'].where(~(df_standard.TAVG_3 > 0) & (df_standard.TAVG_3 < 0.0001), other = 0, inplace =True)
df_standard['TAVG_12'].where(~(df_standard.TAVG_12 > 0) & (df_standard.TAVG_12 < 0.0001), other = 0, inplace =True)
df_standard['TMAX_3'].where(~(df_standard.TMAX_3 > 0) & (df_standard.TMAX_3 < 0.0001), other = 0, inplace =True)
df_standard['TMAX_12'].where(~(df_standard.TMAX_12 > 0) & (df_standard.TMAX_12 < 0.0001), other = 0, inplace =True)

df_standard['SOILMOISTURE_12'].where(~(df_standard.SOILMOISTURE_12 > 0) & (df_standard.SOILMOISTURE_12 < 0.0001), other = 0, inplace =True)
df_standard['SOILMOISTURE_3'].where(~(df_standard.SOILMOISTURE_3 > 0) & (df_standard.SOILMOISTURE_3 < 0.0001), other = 0, inplace =True)

df_standard['WINDSPEED_3'].where(~(df_standard.WINDSPEED_3 > 0) & (df_standard.WINDSPEED_3 < 0.0001), other = 0, inplace =True)
df_standard['WINDSPEED_12'].where(~(df_standard.WINDSPEED_12 > 0) & (df_standard.WINDSPEED_12 < 0.0001), other = 0, inplace =True)

df_standard['PDSI_3'].where(~(df_standard.PDSI_3 > 0) & (df_standard.PDSI_3 < 0.0001), other = 0, inplace =True)
df_standard['PDSI_12'].where(~(df_standard.PDSI_12 > 0) & (df_standard.PDSI_12 < 0.0001), other = 0, inplace =True)

df_standard['VCI_3'].where(~(df_standard.VCI_3 > 0) & (df_standard.VCI_3 < 0.01), other = 0, inplace =True)
df_standard['VCI_12'].where(~(df_standard.VCI_12 > 0) & (df_standard.VCI_12 < 0.0001), other = 0, inplace =True)
df_standard['TCI_3'].where(~(df_standard.TCI_3 > 0) & (df_standard.TCI_3 < 0.0001), other = 0, inplace =True)
df_standard['TCI_12'].where(~(df_standard.TCI_12 > 0) & (df_standard.TCI_12 < 0.0001), other = 0, inplace =True)

df_standard['VHI_3'].where(~(df_standard.VHI_3 > 0) & (df_standard.VHI_3 < 0.0001), other = 0, inplace =True)
df_standard['VHI_12'].where(~(df_standard.VHI_12 > 0) & (df_standard.VHI_12 < 0.0001), other = 0, inplace =True)
df_standard['SPI_3'].where(~(df_standard.SPI_3 > 0) & (df_standard.SPI_3 < 0.0001), other = 0, inplace =True)
df_standard['SPI_12'].where(~(df_standard.SPI_12 > 0) & (df_standard.SPI_12 < 0.0001), other = 0, inplace =True)

print(df_standard.head(5))


# ***

# 
# 
# 
# ## Start of Analysis with cleaned dataset
# 
# 
# ### SQ 1: Do certain variables show correlations and dependencies between each other? 
# 
# This research questions aims to understand underlying relationships between variables in our dataset and to define important correlations and possible causalities, that enhance further approaches in the following research questions

# ### 2. Exploratory Data Analysis
# 
# As a first step classical techniques of Exploratory Data Analysis (EDA) were conducted.
# These techniques include graphical representation of key variables like histograms, line or boxplots.

# #### Graphics of key variables

# In[ ]:


# # Line Graph

# #subset of interesting columns for line plot
# dflines=df_clean[['year', 'yield_kgha', 'district']]

# plt.figure(figsize=(30,10))
# plt.title('Crop yields through years',fontsize=20)
# sns.lineplot(data=dflines, x ='year', y='yield_kgha', hue ='district', palette = 'tab20c')
# plt.legend(bbox_to_anchor=(1.0, -0.15))
# plt.xticks(rotation=90)
# plt.show()

# # Saving the figure on harddrive

# plt.savefig('lines.png', dpi =300)


# In[ ]:


# Line graph of indices

palette=sns.set_palette("Spectral")

#subset of interesting columns for line plot

dflines2=df_clean[['year', 'district','NAOI_3', 'NAOI_12', 'SOI_3', 'SOI_12', 'EVI_3', 'EVI_12', 
               'NDVI_3', 'NDVI_12', 'NDWI_3', 'NDWI_12', 'VCI_3', 'VCI_12', 
               'TCI_3', 'TCI_12', 'VHI_3', 'VHI_12', 'SPI_3', 'SPI_12']]

plt.figure(figsize=(12,6))
plt.title('Time development NAOI',fontsize=15)
sns.lineplot(data=dflines2, x ='year', y='NAOI_3', palette=palette)
sns.lineplot(data=dflines2, x ='year', y='NAOI_12')
plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# Line graph of indices

plt.figure(figsize=(12,6))
plt.title('Time development SOI',fontsize=15)
sns.lineplot(data=dflines2, x ='year', y='SOI_3')
sns.lineplot(data=dflines2, x ='year', y='SOI_12')
plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# Boxplot of several variables

df_boxplot=df_clean[['NAOI_3', 'NAOI_12', 'SOI_3', 'SOI_12', 'EVI_3', 'EVI_12', 
               'NDVI_3', 'NDVI_12', 'NDWI_3', 'NDWI_12', 'VCI_3', 'VCI_12', 
               'TCI_3', 'TCI_12','VHI_3', 'VHI_12', 'SPI_3', 'SPI_12']]

plt.figure(figsize=(15,18))
plt.title('Important Indices',fontsize=20)
sns.boxplot(data=df_boxplot,orient='h', palette = 'binary')
plt.xticks(rotation=0)
plt.show()


#plt.savefig('boxplot_productivity.png', dpi =400)


# In[ ]:


#explore dataframe using the seaborn pairplot function

sns.pairplot(df_boxplot)


# #### Histogramms of variables

# In[ ]:


plt.figure(figsize=(12,10))

# save the histogram data so we can check the bin sequence

n, bins, patches = plt.hist(df_clean['yield_kgha'],bins=30,color='lightgrey')
plt.xlabel('yield_kgha')
plt.ylabel('frequency')
plt.title('Crop yields in kg/ha',fontsize=20)
plt.show()
print('values of histogram: {} \nbin sequence: {}'.format(n, bins))


# In[ ]:


#Scatter Plots

### makes a scatterplot for x and y

plt.figure(figsize=(10,10))
plt.title("Relation of Production and Import",fontsize= 20)
y=df_clean['yield_kgha']
x=df_clean['import']
plt.scatter(x,y,c='grey',marker='o')
plt.ylim[(0, 500)]
plt.ylabel('Crop production')
plt.xlabel('Import of crops')
plt.axis('equal')
plt.show()


# #### Creation of a correlation matrix with all variables, r²

# In[ ]:


# Print a correlation heatmap of the cleaned dataset using all variables to get an overview on the linear correlations

corr= df_standard.corr()
sns.pairplot
f, ax = plt.subplots(figsize=(80, 80))
sns.heatmap(round(corr,2), center=0,cmap=plt.get_cmap('vlag'), robust = True, linecolor='grey', cbar=False,
            square=True, linewidths=.25, annot=True, vmin=-1, vmax=1, annot_kws={"fontsize":30})

#Saving the correlation matrix on harddrive

plt.savefig('corr.png', dpi =300)


# ### 3. Ordinary Least Square Analysis

# We use the Ordinary Least Square Analysis as multiple linear regression model <br>
# We use the implemented OLS in scipy, see documentation here: https://scipy-cookbook.readthedocs.io/items/OLS.html

# In[ ]:


modelols = ols("yield_kgha ~ SPI_12 + SOI_12 + SOILMOISTURE_12 + NAOI_12 + PDSI_12 + TCI_12 ", df_standard).fit()
print(modelols.summary())


# #### Check on multicollinearity of variables: VIF

# ###### Variance Inflation Factor
# 
# The value for VIF starts at 1 and has no upper limit. A general rule of thumb for interpreting VIFs is as follows:
# 
# A value of **1 indicates there is no correlation** between a given explanatory variable and any other explanatory variables in the model.
# 
# A value **between 1 and 5 indicates moderate correlation** between a given explanatory variable and other explanatory variables in the model, but this is often not severe enough to require attention.
# 
# A value **greater than 5 indicates potentially severe correlation** between a given explanatory variable and other explanatory variables in the model. In this case, the coefficient estimates and p-values in the regression output are likely unreliable.
# 
# (see: https://www.statology.org/how-to-calculate-vif-in-python/)

# In[ ]:


#find design matrix for linear regression model using 'rating' as response variable

y, X = dmatrices('yield_kgha ~ SPI_12 + SOI_12 + SOILMOISTURE_12 + NAOI_12 + PDSI_12 + TCI_12 ', 
                 data=df_standard, return_type='dataframe')

#calculate VIF for each explanatory variable

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

#view VIF for each explanatory variable 

vif


# ### 4. Polynomial regression
# 
# Linear regression models are easier to interpret, but polynomial regression models with multiple independent variables can reveal non-linear relationships
# 
# 
# the degree defines the kind of fit we are using, degree =2 refers to quadratic fitting
# if the degree is higher than 2 the fitting is applied on more than one parabolic curves

# In[18]:


variables = df_standard['PRECIPITATION_12']
variables = variables.to_numpy()
variables = variables.reshape((-1,1))
results = df_standard['SPI_12']

# fitting the model called 'poly' with input called 'variables'

poly = PolynomialFeatures(degree=5)
poly_variables = poly.fit_transform(variables)

# splitting in training and testing data to proof if the model is appropriate

poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, results, train_size = 0.5, random_state = 4)

regression = linear_model.LinearRegression()

model = regression.fit(poly_var_train, res_train)
score = model.score(poly_var_test, res_test)

print(score)


# ***

# 
# 
# 
# 
# ### SQ 2: Is it possible to classify the variables into drought periods based on Regression analysis or MachineLearning?

# In[ ]:


# See if there are any statistical differences between drought periods or normal periods

df_clean.groupby('drought_botsw').describe()


# In[ ]:


# See if there are any statistical differences between regions

df_clean.groupby('district').describe()


# In[ ]:


# # See if there are any statistical differences between regions and between periods

drought_each_region= df_clean.groupby(['drought_emdat','district'])

drought_each_region.last()


# ### 5. Predict drought class using Logistic Regression and Naive Bayes classifier 

# #### Logistic Regression

# In[ ]:


# Setting up the variables X and y to be the input for our regression model

X =pd.DataFrame()
y =pd.DataFrame()


# Selecting interesting variables that are optimal for the classification (depends on grouped table)

X = df_standard[['TCI_12', 'SOI_12', 'SPI_12']]
#print(X)
y = df_clean['drought_emdat']
#print(y)


# In[ ]:


# Find out how skewed the data is

numeric_data=X.select_dtypes(include=[np.number])


skewed = X.apply(lambda x: skew(x.dropna().astype(float)))
skewed

X.skew()


# In[ ]:


# If the data shows a skewness of 0.75 the variable gets selected 
# and printed in a histogram before and after logarithmic transformation

skewed = skewed[(skewed > 0.75)]
skewed = skewed.index

X[skewed].hist(bins=20,figsize=(15,2), color='lightblue',xlabelsize=0,ylabelsize=0,grid=False, layout=(1,6))
plt.show()

X[skewed] = np.log1p(X[skewed])
X[skewed].hist(bins=20,figsize=(15,2), color='lightblue',xlabelsize=0,ylabelsize=0,grid=False, layout=(1,6))
plt.show()


# In[ ]:


# A Scaler based on the Scikit StandardScaler is created and applied to X

scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)


# In[ ]:


# Splitting into testing and training data

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7)


# In[ ]:


# Create logistic regression model

LR= LogisticRegression()

#model fitting on training data

LR.fit(X_train, y_train)


# In[ ]:


# report accuracy score for logistic regression model

y_pred=LR.predict(X_test)

print(type(y_pred))
print ('RMSE: ',mean_squared_error(y_test,y_pred,squared=False))
print ('MSE: ',mean_squared_error(y_test,y_pred))
print ('R2: ', r2_score(y_test,y_pred))


# In[ ]:


# How to validate a logistic regression classification?

print('Accuracy: {:.2f}'.format(LR.score(X_test, y_test)))


# #### Classification with Naive Bayes

# In[21]:


# Creating two dataframes containing input variables (A) and class to be determined (b)

A = df_clean[['SOI_12','TCI_12']].to_numpy()


b = df_clean['drought_emdat'].to_numpy()
print(type(b))


# In[22]:


# Splitting into testing and training data

A_train, A_test, b_train, b_test = train_test_split(A,b, train_size=0.7)


# In[23]:


#Selection of Gaussian Naive Bayes Classifier
classifier= GaussianNB()


# In[24]:


b_pred = classifier.fit(A_train, b_train)

print(b_pred)


# In[25]:


#Accuracy of trainig data
accuracy_score(b_train, classifier.predict(A_train))


# In[26]:


# Accuracy of testing data, percentage of correct predictions
#test_scaled = scaler.transform(A_test)
accuracy_score(b_test, classifier.predict(A_test))


# In[28]:


print('Accuracy: {:.2f}'.format(classifier.score(A_test, b_test)))


# ### Research Question 3: Which information and algorithms are needed to predict reliably the crop production? 

# For RQ3 there is the need to predict a numerical values (productivity in our case). Several algortihms are applied and different input variables are tested in their reliability to find the variables that are best used for predicting the productivity

# ### 6. Predicting productivity with Random Forest

# In[ ]:


# Setting the input variables C and the values to be predicted d

C = df_standard[['TAVG_12', 'TAVG_12']]
d = df_standard['yield_kgha']


# In[ ]:


# Checking on the data type of C for further processing

type(d)


# In[ ]:


# Splitting into testing and training data, 80% training data

C_train, C_test, d_train, d_test = train_test_split(C, d, test_size=0.2, random_state = 1604)


# In[ ]:


#d_train.shape

d_train.to_numpy()


# In[ ]:


# We create a Random Forest models with different parameter settings

rf_model = RandomForestRegressor(n_estimators= 100)

# Alternative models

rf_model2 = RandomForestRegressor(n_estimators = 1000, criterion = 'mse', max_depth = None, 
                               min_samples_split = 3, min_samples_leaf = 1)

rf_model3 = RandomForestRegressor(n_estimators = 2000, criterion = 'mse', max_depth = 30, 
                               min_samples_split = 2, min_samples_leaf = 2)


# In[ ]:


# Train it with scaled data and target values

rf_model.fit(C_train, d_train)
rf_model2.fit(C_train, d_train)
rf_model3.fit(C_train, d_train)


# In[ ]:


# How did the model perform on training data?

rf_mse = mean_squared_error(d_train, rf_model.predict(C_train))
rf_mae = mean_absolute_error(d_train, rf_model.predict(C_train))


rf_mse2 = mean_squared_error(d_train, rf_model2.predict(C_train))
rf_mae2 = mean_absolute_error(d_train, rf_model2.predict(C_train))

rf_mse3 = mean_squared_error(d_train, rf_model3.predict(C_train))
rf_mae3 = mean_absolute_error(d_train, rf_model3.predict(C_train))


# In[ ]:


#print("Random Forest training 1 mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))

#print("Random Forest training 2 mse = ",rf_mse2," & mae = ",rf_mae2," & rmse = ", sqrt(rf_mse2))

#print("Random Forest training 3 mse = ",rf_mse3," & mae = ",rf_mae3," & rmse = ", sqrt(rf_mse3))


# In[ ]:


# How did our Random Forest perform on test data?

rf_test_mse = mean_squared_error(d_test, rf_model.predict(C_test))
rf_test_rmse = sqrt(rf_test_mse)
rf_test_mae = mean_absolute_error(d_test, rf_model.predict(C_test))

rf_test_mse2 = mean_squared_error(d_test, rf_model2.predict(C_test))
rf_test_rmse2 = sqrt(rf_test_mse2)
rf_test_mae2 = mean_absolute_error(d_test, rf_model2.predict(C_test))

rf_test_mse3 = mean_squared_error(d_test, rf_model3.predict(C_test))
rf_test_rmse3 = sqrt(rf_test_mse3)
rf_test_mae3 = mean_absolute_error(d_test, rf_model3.predict(C_test))


print("Random Forest test mse = ",rf_test_mse," & mae = ",rf_test_mae," & rmse = ", rf_test_rmse)

print("Random Forest test 2 mse2 = ",rf_test_mse2," & mae2 = ",rf_test_mae2," & rmse2 = ", rf_test_rmse2)

print("Random Forest test 3 mse3 = ",rf_test_mse3," & mae3 = ",rf_test_mae3," & rmse3 = ", rf_test_rmse3)


# In[ ]:


print('R square: {:.2f}'.format(rf_model.score(C_test, d_test)))

print('R square 2nd model: {:.2f}'.format(rf_model2.score(C_test, d_test)))

print('R square 3rd model: {:.2f}'.format(rf_model3.score(C_test, d_test)))


# In[ ]:


print('Accuracy RF Model 1',100*max(0,rf_test_rmse))

print('Accuracy RF Model2',100*max(0,rf_test_rmse2))

print('Accuracy RF Model 3',100*max(0,rf_test_rmse3))


# In[ ]:


# The importance of each variable for the models are discussed by printing the relative importance

feature_list = list(df_standard.columns)

# Variable importance Model 1

# Get numerical feature importances
importances = list(rf_model.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[ ]:


# Variable importance Model 2

# Get numerical feature importances
importances2 = list(rf_model2.feature_importances_)

# List of tuples with variable and importance
feature_importances2 = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances2)]

# Sort the feature importances by most important first
feature_importances2 = sorted(feature_importances2, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances2];


# In[ ]:


# Variable importance Model 3

# Get numerical feature importances
importances3 = list(rf_model3.feature_importances_)

# List of tuples with variable and importance
feature_importances3 = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances3)]

# Sort the feature importances by most important first
feature_importances3 = sorted(feature_importances3, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances3];


# ***

# ### Research Question 4: Do thresholds exist in the dataset that represent breaking points or segments needed for the establishment of a DEWS?
# 
# It is necessary to understand the dynamics of crop yields related to the variables that are forecasted in medium and long-term.
# We try to understand change points in the dataset.

# In[ ]:


dmedian= df_clean.groupby(['drought_emdat','district']).median()


# In[ ]:


dmedian


# ### 7. Change point analysis
# 
# The change point analysis is done by using the ruptures library
# 
# https://github.com/dataman-git/codes_for_articles/blob/master/Change%20point%20detection.ipynb

# In[ ]:


# Section 0: Generate time series

# Sort the dataframe according the years

timeline= df_national.sort_values(by = 'year')

# Varying variance, this variable is changing along time
ts1 = timeline['yield_nation'].values.reshape(-1,1)
ts2 = timeline['SOI_12'].values.reshape(-1,1)
    
plt.figure(figsize=(16,4))
plt.title('Example 2: Yield variance timeseries')
plt.plot(ts1)


# In[ ]:


# Plot the change points:


def plot_change_points(ts,ts_change_loc):
    plt.figure(figsize=(12,4))
    plt.plot(ts)
    for x in ts_change_loc:
        plt.axvline(x,lw=2, color='red')
        
        


# In[ ]:


# detect the change points using ruptures Pelt

algo = rpt.Pelt(model="l1", min_size=2, jump=1).fit(ts1)
change_location = algo.predict(pen=1)
change_location

print(change_location)

# Plot the change points 

plot_change_points(ts1,change_location)


# In[ ]:


# detect the change points #using ruptures Dynamic Programming

algo2 = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(ts1)
change_location2 = algo.predict(pen=1)
#my_bkps = algo.predict(n_bkps=3)

change_location2

# Plot the change points

plot_change_points(ts1,change_location2)


# *** 
# 

# #### Line plots of indicators and change point lines for behavioural analysis

# In[ ]:


# NAOI variables

plt.figure(figsize=(14,6))
#plt.title('Behaviour at breakpoints',fontsize=20)
plt.plot(df_national['NAOI_3'], c='grey', label='NAOI_3')
plt.plot(df_national['NAOI_12'], c='darkblue',label='NAOI_12')
#plt.plot(df_national2['SOI_3'])

plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# SOI variables

plt.figure(figsize=(14,6))
#plt.title('Behaviour at breakpoints',fontsize=20)
plt.plot(df_national['SOI_3'], c='grey', label='SOI_3')
plt.plot(df_national['SOI_12'], c='darkblue', label='SOI_12')
#plt.plot(df_national2['SOI_3'])

plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# PDSI variables

plt.figure(figsize=(14,6))
plt.plot(df_national['PDSI_3'], c='grey', label='PDSI_3')
plt.plot(df_national['PDSI_12'], c='darkblue',label='PDSI_12')

plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# VCI variables

plt.figure(figsize=(14,6))
plt.plot(df_national['VCI_3'], c='grey', label='VCI_3')
plt.plot(df_national['VCI_12'], c='darkblue',label='VCI_12')


plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# TCI variables

plt.figure(figsize=(14,6))
plt.plot(df_national['TCI_3'], c='grey', label='TCI_3')
plt.plot(df_national['TCI_12'], c='darkblue',label='TCI_12')


plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# VHI variables

plt.figure(figsize=(14,6))
plt.plot(df_national['VHI_3'], c='grey', label='VHI_3')
plt.plot(df_national['VHI_12'], c='darkblue',label='VHI_12')


plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# SPI variables

plt.figure(figsize=(14,6))
plt.plot(df_national['SPI_3'], c='grey', label='SPI_3')
plt.plot(df_national['SPI_12'], c='darkblue',label='SPI_12')


plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=5, color='red', linestyle='--')
plt.axvline(x=10, color='red', linestyle='--')
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=15, color='red', linestyle='--')
plt.axvline(x=17, color='red', linestyle='--')
plt.axvline(x=19, color='red', linestyle='--')
plt.axvline(x=21, color='red', linestyle='--')
plt.axvline(x=23, color='red', linestyle='--')
plt.axvline(x=27, color='red', linestyle='--')
plt.axvline(x=29, color='red', linestyle='--')
plt.axvline(x=33, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')

plt.legend(bbox_to_anchor=(1.0, -0.15))
plt.xticks(rotation=0)
plt.show()


# *** 

# End of the analysis
