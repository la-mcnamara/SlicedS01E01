#############################################################
#    Title: SLICED s01e01                                   #
#   Author: Lauren McNamara                                 #
#  Created: 7/1/2021                                        #
# Modified: 7/1/2021                                        #
#  Purpose: Idenitfy risk variable with high code overlap   #
#           for further review.                             #
#############################################################

####################  Setup  ####################
# import packages
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import freqit
from pandas_profiling import ProfileReport
import math
from sklearn.impute import SimpleImputer

# set working, output directories
workdir = r'/Users/Lauren/Documents/Python/Kaggle/SlicedS01E01'
datadir = workdir+'/data/'
plotout = workdir+'/plots/'
os.chdir(workdir)

# date
fdate = datetime.today().strftime('%Y%m%d')

# colors
# source: https://learnui.design/tools/data-color-picker.html#palette
colordict = {'dblue': '#003f5c'
            ,'lblue': '#444e86'
            ,'purple': '#955196'
            ,'pink': '#dd5182'
            ,'orange': '#ff6e54'
            ,'yellow': '#ffa600'
            ,'gray': '#64666B'
            }

####################  Get Data  ####################
traindf = pd.read_csv(datadir+'train.csv')
testdf = pd.read_csv(datadir+'test.csv')

# view header
traindf.head()
testdf.head()

traindf.info()
testdf.info()

trprofile = ProfileReport(traindf, title="Board Games Train", explorative=True)
trprofile.to_widgets()
trprofile.to_file(workdir+"train_profile_report.html")

teprofile = ProfileReport(traindf, title="Board Games Test", explorative=True)
teprofile.to_widgets()

# do not use: names
# may be able to condense: mechanic, category1-12, designer

# use: num_votes, age (same as year?), owned -- all int
# impute zeroes: min_players, max_players, avg_time, min_time, max_time -- all int
# impute unrealistic min: year --int

# outcome: geek_rating

traindf['geek_rating'].hist()
traindf['log_geek_rating'] = traindf['geek_rating'].apply(lambda x: math.log(x))
traindf[['geek_rating','log_geek_rating']].head()
traindf.log_geek_rating.hist()

####################  Preprocess  ####################

# look at unlikely years
traindf.loc[traindf['year'] < 1000, ['names','year','designer']]
# the games with very early years appear to be accurate (Backgammon, Go have 
# been played since BC for example). Do not impute year.

# cut year into bins
traindf['year_grp'] = pd.qcut(traindf['year'], q=3)
traindf.groupby('year_grp')['year'].describe()
freqit.oneway(traindf['year_grp']).freqtable()

yrcat = pd.get_dummies(traindf['year_grp'], prefix='yearcat')

traindf = traindf.merge(yrcat
                        ,how='outer'
                        ,left_index=True
                        ,right_index=True
                        ,indicator=True)
traindf.head()
traindf._merge.value_counts()

traindf = traindf.rename(columns = {'yearcat_(-3000.001, 2006.0]':'yearcat_1'
                                    ,'yearcat_(2006.0, 2013.0]':'yearcat_2'
                                    ,'yearcat_(2013.0, 2018.0]':'yearcat_3'})

# impute zeroes
def impute_zero_to_mean(series):
    """
    impute zeroes to mean of column
    return column  of imputed values and column of impute flags 
    """
    col = series.to_numpy().reshape(-1,1)
    imputer = SimpleImputer(missing_values=0, strategy='mean', add_indicator=True)
    imputer.fit(col)
    imp_col = imputer.transform(col)
    imputed = pd.Series(imp_col[:,0])
    impflag = pd.Series(imp_col[:,1])
    return imputed, impflag

imp_zero_cols = ['min_players','max_players','avg_time','min_time','max_time']
for i in imp_zero_cols:
    traindf[i+'_imp'], traindf[i+'_impflg'] = impute_zero_to_mean(traindf[i])



####################  Model  ####################