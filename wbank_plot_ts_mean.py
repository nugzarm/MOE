# Purpose: plot time series or box plots
# Input: files produced by wbank_das.py
# Output: plots
# Author: Nugzar Margvelashvili
# Created: May 2023
# Last updated: 9 June 2023

#importing libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
#from mpl_toolkits.basemap import Basemap
import os
os.environ['USE_PYGEOS'] = '0'

#import geopandas
#import folium
import sys;
import math;
import matplotlib.colors;
#import seaborn as sns;
#https://www.learnpythonwithrune.org/plot-world-data-to-map-using-python-in-3-easy-steps/

############################
# subsample dfData for one year
def dfDataPerYear(dfData, year):
    df_year = dfData[dfData['Year']==year];
    ## fill blanks with 0
    df_year.loc[:,'Entity'] = df_year['Entity'].replace(" ","");
    return df_year;

##############################
#################################
import itertools

PLOT_TS = +1;
#simList=["sc_DAS_GEO2/fileSimX.csv", "sc_DAS_GEO2/fileSimX.csv_fine"]
#obsList=["sc_DAS_GEO2/fileObsX.csv", "sc_DAS_GEO2/fileObsX.csv_fine"]
#simList2=["sc_DAS_GEO3/fileSimX.csv", "sc_DAS_GEO3/fileSimX.csv_fine"]

obsList=["sc_02_global_2050/fileSimX_.csv", "sc_02_global_2050/fileObsX.csv_fine"]
simList=["sc_02_global_2050/fileSimX_.csv", "sc_02_global_2050/fileSimX.csv_fine"]
simList1=["sc_02_eu_2050/fileSimX_.csv",    "sc_02_eu_2050/fileSimX.csv_fine"]
simList2=["sc_02_easia_2050/fileSimX_.csv", "sc_02_easia_2050/fileSimX.csv_fine"]
simList3=["sc_02_meast_2050/fileSimX_.csv", "sc_02_meast_2050/fileSimX.csv_fine"]
simList4=["sc_02_fback_2050/fileSimX_.csv", "sc_02_fback_2050/fileSimX.csv_fine"]

'''
obsList=["sc_03_seed1/fileObsX.csv", "sc_03_seed1/fileObsX.csv_fine"]
simList=["sc_03_seed1/fileSimX.csv", "sc_03_seed1/fileSimX.csv_fine"]
simList1=["sc_03_CO2_const/fileSimX.csv", "sc_03_CO2_const/fileSimX.csv_fine"]
simList2=["sc_03_war_const/fileSimX.csv", "sc_03_war_const/fileSimX.csv_fine"]
simList3=["sc_03_covid_pulse/fileSimX.csv", "sc_03_covid_pulse/fileSimX.csv_fine"]
simList4=["sc_03_CO2_drop/fileSimX.csv", "sc_03_CO2_drop/fileSimX.csv_fine"]
'''

file_i = 0;
for (obsFile, dataFile) in zip(obsList, simList): 

    # get Observations
    dfObs = pd.read_csv(obsFile, sep=' ', header=0)

    # get simulated data
    dfData = pd.read_csv(dataFile, sep=' ', header=0)
    print(dfData.head())

    dfData1 = pd.read_csv(simList1[file_i], sep=' ', header=0)

    dfData2 = pd.read_csv(simList2[file_i], sep=' ', header=0)
    print(dfData2.head())

    dfData3 = pd.read_csv(simList3[file_i], sep=' ', header=0)
    print(dfData3.head())

    dfData4 = pd.read_csv(simList4[file_i], sep=' ', header=0)
    print(dfData4.head())


    file_i = file_i +1


    #define how to aggregate various fields
    domains = list(dfData.columns.values)
    domains.remove('Year')
    print(domains)
    domains1 = list(dfData1.columns.values)
    domains1.remove('Year')
    domains2 = list(dfData2.columns.values)
    domains2.remove('Year')
    domains3 = list(dfData3.columns.values)
    domains3.remove('Year')
    domains4 = list(dfData4.columns.values)
    domains4.remove('Year')


    agg_mean = dict.fromkeys(domains, 'mean');
    agg_std = dict.fromkeys(domains, 'std');

    agg_mean1 = dict.fromkeys(domains1, 'mean');
    agg_std1 = dict.fromkeys(domains1, 'std');

    agg_mean2 = dict.fromkeys(domains2, 'mean');
    agg_std2 = dict.fromkeys(domains2, 'std');

    agg_mean3 = dict.fromkeys(domains3, 'mean');
    agg_std3 = dict.fromkeys(domains3, 'std');

    agg_mean4 = dict.fromkeys(domains4, 'mean');
    agg_std4 = dict.fromkeys(domains4, 'std');

    # aggregate obs
    df_obs = dfObs.groupby(dfObs['Year']).agg(agg_mean)
    #aggregate model ensemble
    df_mean = dfData.groupby(dfData['Year']).agg(agg_mean)
    df_std = dfData.groupby(dfData['Year']).agg(agg_std)

    df_mean1 = dfData1.groupby(dfData1['Year']).agg(agg_mean1)
    df_std1 = dfData1.groupby(dfData1['Year']).agg(agg_std1)

    df_mean2 = dfData2.groupby(dfData2['Year']).agg(agg_mean2)
    df_std2 = dfData2.groupby(dfData2['Year']).agg(agg_std2)

    df_mean3 = dfData3.groupby(dfData3['Year']).agg(agg_mean3)
    df_std3 = dfData3.groupby(dfData3['Year']).agg(agg_std3)

    df_mean4 = dfData4.groupby(dfData4['Year']).agg(agg_mean4)
    df_std4 = dfData4.groupby(dfData4['Year']).agg(agg_std4)


    print(df_obs.head())
    print(df_mean.head())
    print(df_std.head())

    print("bebebe")
    #print( df_mean.iloc[:,0] )
    print( df_mean.index )

    '''
    df_mean = df_mean.loc[ df_mean.index > 2015]
    df_mean1 = df_mean1.loc[ df_mean1.index > +2015]
    df_mean2 = df_mean2.loc[ df_mean2.index > +2015]
    df_mean3 = df_mean3.loc[ df_mean3.index > +2015]
    df_mean4 = df_mean4.loc[ df_mean4.index > +2015]    
    '''
    print("bububu")
    print(df_mean.head())


    # plot ts of obs, ensemble mean, and shaded stdev
    for i, c in enumerate(df_mean.columns):
        fig, axs = plt.subplots(figsize =(5,5))

        #df_obs[c].plot(ax=axs, lw=2, color="magenta")
        df_mean[c].plot(ax=axs, lw=4, color="red") # Global
        df_mean1[c].plot(ax=axs, lw=2, color="cyan") # EU  CO2
        df_mean2[c].plot(ax=axs, lw=2, color="orange") # EAsia  WAR
        df_mean3[c].plot(ax=axs, lw=2, color="green") # Meast  COVID


        '''
        df_mean[c].plot(ax=axs, lw=4, color="blue") # Baseline
        df_mean1[c].plot(ax=axs, lw=2, color="green") #COVID
        df_mean2[c].plot(ax=axs, lw=2, color="orange") # WAR
        df_mean3[c].plot(ax=axs, lw=2, color="cyan") # CO2
        '''
        
        #df_mean4[c].plot(ax=axs, lw=2, color="blue")
        #df_obs[c].plot(ax=axs, lw=2, color="cyan");
        
        plt.title(c);
        fig.savefig('fig00_'+c+'.jpg')
        plt.show()
        plt.close()




