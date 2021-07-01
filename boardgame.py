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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import freqit

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