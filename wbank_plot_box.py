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


PLOT_TS = -1;
if(PLOT_TS > 0):
    simList=["fileSimX.csv_fine",  "fileSimX_.csv"]
    obsList=["fileObsX.csv_fine",  "fileObsX.csv"]
else: # box plots
    simList=["./sc_02_global/fileObsX.csv"]
    obsList=["./sc_02_global/fileObsX.csv"]



for (obsFile, dataFile) in zip(obsList, simList): 

    # get Observations
    #obsFile = "fileObs.csv_fine"
    #obsFile = "fileObs.csv"
    dfObs = pd.read_csv(obsFile, sep=' ', header=0)

    # get simulated data
    # baseline
    #dataFile = "fileSim.csv_fine"
    #dataFile = "fileSim.csv"
    dfData = pd.read_csv(dataFile, sep=' ', header=0)
    print(dfData.head())

    #define how to aggregate various fields
    domains = list(dfData.columns.values)
    domains.remove('Year')
    print(domains)

    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for c in [c for c in dfObs.columns if dfObs[c].dtype in numerics and c != 'Year']:
        dfObs[c] = np.log(dfObs[c].abs())
    for c in [c for c in dfData.columns if dfData[c].dtype in numerics and c != 'Year']:
        dfData[c] = np.log(dfData[c].abs())
    """

    #agg_mean = {'envi': 'mean', 'agri': 'mean', 'econ': 'mean', 'tech': 'mean', 'soci': 'mean', 'conf': 'mean', 'all': 'mean'}
    #agg_std = {'envi': 'std', 'agri': 'std', 'econ': 'std', 'tech': 'std', 'soci': 'std', 'conf': 'std', 'all': 'std'}

    agg_mean = dict.fromkeys(domains, 'mean');
    agg_std = dict.fromkeys(domains, 'std');

    # aggregate obs
    df_obs = dfObs.groupby(dfObs['Year']).agg(agg_mean)
    #aggregate model ensemble
    df_mean = dfData.groupby(dfData['Year']).agg(agg_mean)
    df_std = dfData.groupby(dfData['Year']).agg(agg_std)

    print(df_obs.head())
    print("df_mean.head():")
    print(df_mean.head())
    print("df_std.head()")
    print(df_std.head())



    # plot ts of obs, ensemble mean, and shaded stdev
    if(PLOT_TS > 0):
        for i, c in enumerate(df_mean.columns):
            fig, axs = plt.subplots(figsize =(4,3))

            #color = plt.rcParams['axes.prop_cycle'].by_key()["color"][i]
            color="silver"

            axs.fill_between(df_mean.index, (df_mean - 1e-0*df_std)[c], (df_mean+1e-0*df_std)[c],
                             facecolor=matplotlib.colors.to_rgba(color, .4), edgecolor=color)
            df_mean[c].plot(ax=axs, lw=1)
            #df_obs[c].plot(ax=axs, lw=2);

            #ax1 = axs.twinx()
            # Set the limits of the new axis from the original axis limits
            #ax1.set_ylim(axs.get_ylim())

            df_obs[c].plot(ax=axs, lw=2);


            plt.title(c);
            fig.savefig('fig00_'+c+'.jpg')
            plt.show()
            plt.close()


######################################################
import seaborn as sns;
# for each domain (envi, agri, econ, ...)
# creates a box plot of aggregated indicators
# for baseline, 75%, 99% and 5% scenarios

# read scenarios
'''
#dataFile0 = "./sc_DAS_TSTglo/fileSimX.csv"
#dfData_0 = pd.read_csv(dataFile0, sep=' ', header=0)
dataFile1 = "./sc_02_global_2050/fileSimX_.csv"
dfData_1 = pd.read_csv(dataFile1, sep=' ', header=0)
dataFile2 = "./sc_02_eu_2050/fileSimX_.csv"
dfData_2 = pd.read_csv(dataFile2, sep=' ', header=0)
dataFile3 = "./sc_02_easia_2050/fileSimX_.csv"
dfData_3 = pd.read_csv(dataFile3, sep=' ', header=0)
dataFile4 = "./sc_02_meast_2050/fileSimX_.csv"
dfData_4 = pd.read_csv(dataFile4, sep=' ', header=0)
#dataFile5 = "./sc_02_meast_2050/fileSimX_.csv"
#dfData_5 = pd.read_csv(dataFile5, sep=' ', header=0)
'''

dataFile1 = "./sc_03_seed1/fileSimX.csv"
dfData_1 = pd.read_csv(dataFile1, sep=' ', header=0)
dataFile2 = "./sc_03_CO2_const/fileSimX.csv"
dfData_2 = pd.read_csv(dataFile2, sep=' ', header=0)
dataFile3 = "./sc_03_war_const/fileSimX.csv"
dfData_3 = pd.read_csv(dataFile3, sep=' ', header=0)
dataFile4 = "./sc_03_covid_pulse/fileSimX.csv"
dfData_4 = pd.read_csv(dataFile4, sep=' ', header=0)
dataFile5 = "./sc_03_CO2_drop/fileSimX.csv"
dfData_5 = pd.read_csv(dataFile5, sep=' ', header=0)





'''
dataFile0 = "./sc_DAS_TSTglo/fileSimX.csv"
dfData_0 = pd.read_csv(dataFile0, sep=' ', header=0)
dataFile1 = "./sc_DAS_MLIglo/fileSimX.csv"
dfData_1 = pd.read_csv(dataFile1, sep=' ', header=0)
dataFile2 = "./sc_DAS_PHLglo/fileSimX.csv"
dfData_2 = pd.read_csv(dataFile2, sep=' ', header=0)
dataFile3 = "./sc_DAS_GEOglo/fileSimX.csv"
dfData_3 = pd.read_csv(dataFile3, sep=' ', header=0)
dataFile4 = "./sc_DAS_AUSglo/fileSimX.csv"
dfData_4 = pd.read_csv(dataFile4, sep=' ', header=0)
dataFile5 = "./sc_DAS_FKTglo/fileSimX.csv"
dfData_5 = pd.read_csv(dataFile5, sep=' ', header=0)
'''

#domains = list(dfData_1.columns.values)

# create box-plots
# loop over domains (envi, agri, ...)
if(PLOT_TS < 0):
    box_colors = {'A': 'purple', 'B': 'pink', 'C': 'gold'}
    for i, c in enumerate(df_mean.columns):
        print(c+'3')
        cc  = c + ""
        cc0 = cc+'0'
        cc1 = cc+'1'
        cc2 = cc+'2'
        cc3 = cc+'3'
        cc4 = cc+'4'
        cc5 = cc+'5'
        
        fig, axs = plt.subplots(figsize =(5,5))
        #dfData_colx = dfObs.loc[:,[cc]]
        #dfData_col = dfData_colx.rename(columns={cc:cc0}, inplace=False)

        dfData_col = pd.DataFrame()
        #dfData_col[cc0] = dfData_0[cc].copy()
        dfData_col[cc1] = dfData_1[cc].copy()
        dfData_col[cc2] = dfData_2[cc].copy()
        dfData_col[cc3] = dfData_3[cc].copy()
        dfData_col[cc4] = dfData_4[cc].copy()
        dfData_col[cc5] = dfData_5[cc].copy()
        
        box_colors = {cc0: 'red', cc1: 'blue', cc2: 'aqua',  cc3: 'gold', cc4: 'limegreen', cc5: 'grey'}

        ax = sns.boxplot(data=dfData_col, showfliers = False, palette=box_colors, whis=[5,95])
        plt.savefig('fig_box100_'+c+'.jpg')
        plt.show()
        plt.close()
        #print(dfData_col.head(33));
        #print(dfObs.head(33));
        #print(dfData_col[cc5]);
        #print(dfData_5[cc]);
        #print(dfData_5.head(33));
        #sys.exit()


sys.exit()



"""
# box plots on a sungle graph (not used)

fig, axs = plt.subplots(figsize =(8,5))
dfData_val = dfData.drop(['Year'], axis=1).assign(Scenario=0)
dfData_val_1 = dfData_1.drop(['Year'], axis=1).assign(Scenario=1)
dfData_val_2 = dfData_2.drop(['Year'], axis=1).assign(Scenario=2)
dfData_val_3 = dfData_3.drop(['Year'], axis=1).assign(Scenario=3)

cdf = pd.concat([dfData_val, dfData_val_1, dfData_val_2, dfData_val_3])
mdf = pd.melt(cdf, id_vars=['Scenario'], var_name=['Letter'])
print(mdf.head())
ax = sns.boxplot(x="Letter", y="value", hue="Scenario", data=mdf, showfliers = False)
plt.savefig('fig_box.jpg')
plt.show()
plt.close()
"""


sys.exit()






# subsample a year to start
df_init = dfDataPerYear(dfData, YEAR_START);

if VERBOSE == 1:
    print(df_init.head())
    print("df_init.shape()=: ", df_init.shape)

df_sum = df_init.copy()
df_sumf = df_init.copy()

#titles = df_init.iloc[0].tolist()
titles = list(df_init.columns)
if VERBOSE == 1:
    print("titles_all: ", titles)
#rab1.remove('LN')
titles.pop(0)
titles.pop(0)
titles.pop(0)
titles.pop(0)
codes = df_init["Code"]
entities = df_init["Entity"]
if VERBOSE == 1:
    print("")
    print("titles cut: ", titles)
    print("codes: ", codes)
    print("entities: ", entities)

domIDs=[]
scales=[]
for title in titles:
    items = title.split('_')
    domIDs.append(items[0])
    scales.append(items[1])
for i in range (len(scales)):    
    if( int(scales[i]) <1 ):
        scales[i]=-1
if VERBOSE == 1:
    print('domIDs')
    print(domIDs)
    print('scales')
    print(scales)

df_init = df_init.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
df_init = df_init.replace(np.nan,0)
np_init = df_init.to_numpy()

np_init0 = np.copy(np_init)

#np.random.seed(42)
#df = pd.DataFrame(np.random.randn(600, 5),columns=['a', 'b', 'c', 'd', 'e'])
#df_corr = df.corr()

dfData_year = dfDataPerYear(dfData, year_f);
dfData_year = pd.merge(dfData_year, codes, how="right", on=["Code"])
dfData_year = dfData_year.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
dfData_year = dfData_year.replace(np.nan,0)
npData_year = dfData_year.to_numpy()

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
    
    scoring(df_sumf, npData_year, domains, domIDs);


# plot
df_init.plot()
plt.show()


def folium_plot(df_init, df_sum, titles):

    # plot
    # Read the geopandas dataset
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    print(world.head())
    world.head()
    df_init = world.merge(df_init, how="left", left_on=['iso_a3'], right_on=['Code'])
    df_init['name']=df_init['name'].str.replace(" ","")

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
    i=0;
    my_map2=[]
    for nameVar in domains:
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
            fill_color='YlOrRd',
            nan_fill_color="White",
            fill_opacity=0.71,
            line_opacity=0.2,
            legend_name=nameVar
        ).add_to(my_map2[i])
        my_map2[i].save('net_'+nameVar+'.html')
        i = i + 1

folium_plot(df_init, df_sum, titles)

