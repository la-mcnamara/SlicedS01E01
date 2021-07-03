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

# use: num_votes, age (same as year?), owned
# impute zeroes: min_players, max_players, avg_time, min_time, max_time
# impute unrealistic min: year

# outcome: geek_rating