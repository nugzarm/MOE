# Purpose: plot state level variables on the globe
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

import geopandas
import folium
import branca.colormap as cm

import sys;
import math;

#https://www.learnpythonwithrune.org/plot-world-data-to-map-using-python-in-3-easy-steps/


############################
# subsample dfData for one year
#def dfDataPerYear(dfData, year):
##    df_year = dfData[dfData['Year']==year];
    ## fill blanks with 0
#    df_year.loc[:,'Entity'] = df_year['Entity'].replace(" ","");
#    return df_year;


def dfDataPerYear(dfData2, year):
    #print("up here");
    df_year = (dfData2.loc[dfData2['Year']==year]).copy();
    #sys.exit()
    ## get rid of blanks
    #df_year.loc[:,'Entity'] = df_year['Entity'].replace(" ","");

    df_col = (df_year.Entity.replace(' ', '',regex = True) ).copy()
    df_year['Entity'] = df_col.copy()

    return df_year;


##############################
#################################

YEAR_START = 1991
VERBOSE = 0
domains=["pollutant", "agriculture", "economy", "technology", "social", "conflict", "all"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# initial data
# read all data
dataFile = "dfData.csv"
#dataFile = "fileSim_85x.csv"
dfData = pd.read_csv(dataFile, sep=',', header=0)

#scoreFile = "dfScore.csv"
#scoreFile = "./sc_000_bsln2/df_sum.csv"
scoreFile = "./df_sum.csv"

dfScore = pd.read_csv(scoreFile, sep=',', header=0)

print("dfScore.head()")
print(dfScore.head())

#dfData_year = dfDataPerYear(dfData, year);
#dfData_year = pd.merge(dfData_year, codes, how="right", on=["Code"])
#dfData_year = dfData_year.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
#dfData_year = dfData_year.replace(np.nan,0)
#npData_year = dfData_year.to_numpy()

# subsample a year to start
dfData_year = dfDataPerYear(dfData, YEAR_START);
dfScore_year = dfDataPerYear(dfScore, YEAR_START);

dfScore_year = dfScore_year.replace(np.nan,0)

print(dfScore_year.head())
#sys.exit()

# replace empty cells with column-mean
for c in [c for c in dfScore.columns if dfScore[c].dtype in numerics and c != 'Year']:
    #x = dfData[c].mean()
    x = dfScore[c].median()
    dfScore[c].fillna(x, inplace = True)



if VERBOSE == 1:
    print(dfData_year.head())
    print("dfData_year.shape()=: ", dfData_year.shape)

#titles = df_init.iloc[0].tolist()
titles = list(dfData_year.columns)
if VERBOSE == 1:
    print("titles_all: ", titles)
#rab1.remove('LN')
titles.pop(0)
titles.pop(0)
titles.pop(0)
titles.pop(0)
codes = dfData_year["Code"]
entities = dfData_year["Entity"]



#df_init = df_init.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
#df_init = df_init.replace(np.nan,0)
#np_init = df_init.to_numpy()



#dfData_year = dfDataPerYear(dfData, YEAR_START);
#dfData_year = pd.merge(dfData_year, codes, how="right", on=["Code"])
#dfData_year = dfData_year.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
#dfData_year = dfData_year.replace(np.nan,0)
#npData_year = dfData_year.to_numpy()

"""
# get observations
for year in range(YEAR_START, YEAR_STOP):
    if year < 2020:
        year_f = year
    else:
        year_f = 2019
    #print("year_f= ",year_f)
    # forcing / observations data
    dfData_year = dfDataPerYear(dfData, year_f);
    dfData_year = dfData_year.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
    dfData_year = dfData_year.replace(np.nan,0)
    npData_year = dfData_year.to_numpy()
"""    


def folium_plot(df_init, df_sum, titles):
    # Read the geopandas dataset
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est>0) & (world.name!="Antarctica")] 
    print(world.head())
    world.head()
    df_init = world.merge(df_init, how="left", left_on=['iso_a3'], right_on=['Code'])
    df_init['name']=df_init['name'].str.replace(" ","")

    print("df_init.head()")
    print(df_init.head())

    # Create a map
    i=0;
    my_map=[]
    for nameVar in titles:
        print(i, nameVar)
        my_map.append( folium.Map() )
        # Add the data
        folium.Choropleth(
            geo_data=df_init,
            name='choropleth',
            data=df_init,
            columns=['name', nameVar],
            key_on='feature.properties.name',
            fill_color='YlOrRd',
            nan_fill_color="White",
            fill_opacity=0.71,
            line_opacity=0.2,
            legend_name=nameVar
        ).add_to(my_map[i])
        my_map[i].save('out_'+nameVar+'.html')
        i = i + 1
    
    # print net points
    df_sum = world.merge(df_sum, how="left", left_on=['iso_a3'], right_on=['Code'])
    df_sum['name']=df_sum['name'].str.replace(" ","")
    print("df_sum.head()")
    print(df_sum.head())

    #linear = cm.LinearColormap(["green", "yellow", "red"], vmin=df_sum['segment'].min(), vmax=df_sum['segment'].max())
    #m = folium.Map([df['lat'].mean(), df['lon'].mean()], tiles="cartodbpositron", zoom_start=2)


    #step = cm.StepColormap( ['yellow', 'green', 'purple'], vmin=3, vmax=10,
    #    index=[3, 6, 8, 10],  #for change in the colors, not used fr linear
    #    caption='Color Scale for Map'    #Caption for Color scale or Legend
    #)



    i=0;
    my_map2=[]
    for nameVar in domains:

        #linear = cm.LinearColormap(["green", "yellow", "red"], vmin=df_sum[nameVar].min(), vmax=df_sum[nameVar].max())
        #m = folium.Map([df['lat'].mean(), df['lon'].mean()], tiles="cartodbpositron", zoom_start=2)
        vmin=df_sum[nameVar].min(); 
        vmax=df_sum[nameVar].max();
        myColor="RdYlBu_r"


        if nameVar == "all":            
            vmin = 0.18;
            vmax = 0.39;

        if nameVar == "pollutant":
            vmin = -0.97;
            vmax = -0.34;
            myColor="RdGy_r"


        vmin = -1.1;
        vmax = +1.1;



        delta = (vmax - vmin) / 10.0;

        my_map2.append( folium.Map() )
        # Add the data
        folium.Choropleth(
            geo_data=df_sum,
            name='choropleth',
            #data=happy.loc[(happy['Year']=='2018')],
            data=df_sum,
            #columns=['Entity', 'GDP_2015_USD'],
            columns=['name', nameVar],
            key_on='feature.properties.name',
            bins=[vmin-0.0001, vmin+delta, vmin+2*delta, vmin+3*delta, vmin+4*delta,vmin+5*delta,vmin+6*delta,vmin+7*delta, vmin+8*delta,vmin+9*delta,vmax+0.0001],
            #fill_color='YlOrRd',
            #fill_color="PRGn",
            fill_color=myColor,
            #fill_color="RdYlGn",
            #fill_color="RdYlBu_r",
            nan_fill_color="White",
            fill_opacity=0.71,
            line_opacity=0.2,
            legend_name=nameVar
        ).add_to(my_map2[i])
        my_map2[i].save('net_'+nameVar+'.html')
        i = i + 1
    
folium_plot(dfData_year, dfScore_year, titles)

