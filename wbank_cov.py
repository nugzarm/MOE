# Purpose: calculate covariance matrix and dump it to disk
# Input: "Our World in Data" styled data
# Example:
#   Entity,Code,Year,2_1_GDP_2015_USD
#   Afghanistan,AFG,2002,7228792320
#   Afghanistan,AFG,2003,7867259392
# Output: covariance matrix: "cova.csv"
# Author: Nugzar Margvelashvili
# Created: May 2023
# Last updated: 9 June 2023


#https://www.learnpythonwithrune.org/plot-world-data-to-map-using-python-in-3-easy-steps/
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

#fname = ""
#file_x='states.csv'
#fx = open(file_x);
file_states='states.csv'
states = pd.read_csv(file_states, sep=',', header=0)
print(states.head())

# LINE, LOGA, LOGI, SQRT, INVE
TRANSF = "LOGA"
#TRANSF = "LINE"

scale = 1.
YEAR_START = 1970
YEAR_STOP  = 2017

domains=["pollutant", "agriculture", "economy", "technology", "social", "conflict"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


##############################################  USER INPUT
##############################################

# Select a set of countries to calculate covariance
datadir = "world_data_global/"
#datadir = "world_data_eu/"
#datadir = "world_data_meast/"
#datadir = "world_data_axis/"
#datadir = "world_data_easia/"

##############################################
############################################## end USER INPUT


colNames=[
        #Surface temperature anomaly, 2017
        #Surface temperature anomaly, measured in degrees Celsius The temperature anomaly is
        #relative to the 1951-1980 global average temperature. Data is based on the HadCRUT analysis
        #from the Climatic Research Unit (University of East Anglia) in conjunction with the Hadley
        "0_0_temp_anomaly", 
        #"0_0_CO2pc",
        "0_0_CO2", #"0_0_ShareOfGlobalCO2", 
        "0_0_NitroPerHa", 
        "0_0_PhosPerHa",
        "0_0_PotashPerHa",
        #"0_0_Plastic_t_y", 
        "0_0_Pesticides_kg_ha", 
        "0_0_Fertilizer_kg_ha", #2024#
        #"0_0_PM25", "0_0_DisasterDeath",

        "1_1_LandArea_sq_km", 
        "1_1_Forest_share", #"1_1_ForestArea", 
        "1_1_ArableLand_ha_pop",
        #2024#"1_1_Pesticides_kg_ha", "1_1_Fertilizer_kg_ha",
        "1_1_CerealsYield_t_ha", #"1_1_Water_withdrawal_pc",

        "2_1_GDP_2015_USD", 
        "2_1_GDP_Growth",
        "2_1_GDPPC2015", 
        "2_1_GDPPC_Growth",        
        "2_1_Population",
        "2_1_Population_density",
        # Shown is the 'trade openness index'- the sum of exports and imports of goods and services, divided by the gross domestic product (0 : 500%).
        "2_1_Trade_share_GDP",
        #Foreign direct investment, net outflows as share of GDP, 2020
        #Foreign direct investment (FDI) refers to direct investment equity flows in an economy. It is the sum
        #of equity capital, reinvestment of earnings, and other capital. This series shows net outflows of
        #investment from the reporting economy to the rest of the world, and is divided by GDP (-5 : +5 %)       
        # nm: in other words, it shows what fraction of GDP residents of this country have invested into businesses operating/residing in other countries 
        "2_1_FDI_outflows_share_GDP", 
        "2_1_Military_usd_pc",
        "2_1_EnergyPC", 
        
        #"3_0_Energy_intensity",
        #"3_1_SchoolYears",
        "3_1_Research_frac_GDP",
        "3_1_Researchers_per_million",
        #"3_1_Renewables_kWpc",
        "3_1_Renewables_electricity",

        #"4_1_Migrant_pcent", 

        #"4_1_ExpSchoolYears",
        "4_1_SchoolYears", 
        "4_1_LifeExpect",
        #"4_1_civlibs_score_fh", 
        "4_1_Libdem", 
        "4_1_hdi_index",
        #"4_1_Happy", 
        "4_1_PopGrowthRate", 
        "4_0_RichestShare",
        
        "5_0_HomicideDeath", 
        "5_0_BattleDeath"
        ]


logaNot=["0_0_temp_anomaly", "2_1_GDP_Growth", "2_1_GDPPC_Growth",  "2_1_FDI_outflows_share_GDP", "4_1_PopGrowthRate" ]

logaYes = []
for item in colNames:
    if item not in logaNot:
        logaYes.append(item);

if TRANSF == "LINE":
    # logTransform nothing
    logaNot = colNames.copy();
    logaYes = [];

fileNames=[
        datadir+"nm_hadcrut-surface-temperature-anomaly.csv", 
        #"nm_co-emissions-per-capita.csv",
        datadir+"nm_co2-emissions-per-country.csv", # 'nm_annual-share-of-co2-emissions.csv', 
        datadir+"nm_nitrogen-fertilizer-application-per-hectare-of-cropland.csv", 
        datadir+"nm_phosphate-application-per-hectare-of-cropland.csv", 
        datadir+"nm_potash-fertilizer-application-per-hectare-of-cropland.csv", 
        #"nm_plastic-waste-mismanaged.csv", ##### poor quolity of dat, short and sparse
        datadir+'nm_pesticide-per-ha-of-cropland.csv', 
        datadir+'nm_fertilizer-per-ha-of-arable-land.csv', #2024#
        #'nm_PM25-air-pollution.csv', #'nm_share-deaths-from-natural-disasters.csv',  ##### poor quality records

        datadir+"nm_land-area-km.csv", 
        datadir+'nm_forest-area-as-share-of-land-area.csv', #'nm_forest-area-km.csv', 
        datadir+'nm_arable-land-use-per-person.csv',
        #2024#'nmnm_pesticide-per-ha-of-cropland.csv', 'nmnm_fertilizer-per-ha-of-arable-land.csv',
        datadir+'nm_cereal-yield.csv', #"nm_water-withdrawals-per-capita.csv",

        datadir+'nm_gross-domestic-product.csv', 
        datadir+'nm_gross-domestic-product_growth_frac.csv',
        #GDP per capita, 2020
        #This data is expressed in US dollars. It is adjusted for inflation but does not account for
        #differences in the cost of living between countries.
        datadir+'nm_gdp-per-capita-in-us-dollar-world-bank.csv',
        datadir+'nm_gdp-per-capita-in-us-dollar-world-bank_growth.csv',        
        datadir+"nm_population-since-1800.csv", 
        datadir+"nm_population-density.csv",
        datadir+"nm_trade-as-share-of-gdp.csv", 
        datadir+"nm_foreign-direct-investment-net-outflows-of-gdp.csv",
        datadir+'nm_military-spending-per-capita.csv',
        datadir+"nm_per-capita-energy-use.csv", 
        
        #"nm_energy-intensity-of-economies.csv", # the record too short 2000 and onwards
        #"nmnm_mean-years-of-schooling-long-run.csv",
        datadir+"nm_research-spending-gdp.csv",
        datadir+"nm_researchers-in-rd-per-million-people.csv",
        #datadir+"nm_per-capita-renewables.csv", ## poor record - missing africa, too short
        datadir+"nm_electricity-renewables_share.csv",
        ###datadir+"nm_electricity-renewables_twt.csv",

        #'nm_migrant-stock-share.csv', 

        #Expected years of schooling, 2021
        #The number of years a child of school entrance age can expect to receive if the current
        #age-specific enrollment rates persist throughout the childs years of schooling.(1990 -2021)
        #"nm_expected-years-of-schooling.csv",

        #Average years of schooling, 2017
        #Average number of years people aged 25+ participated in formal education.(up to 2017)
        datadir+'nm_mean-years-of-schooling-long-run.csv',         
        datadir+"nm_life-expectancy.csv",
        #'nm_civil-liberties-score-fh.csv', 
        datadir+'nm_liberal-democracy.csv', 
        datadir+'nm_human-development-index.csv',
        #'nm_happiness-cantril-ladder.csv', 
        datadir+"nm_population-growth-rates.csv", 
        datadir+'nm_income-share-of-the-top-10-pip.csv', 
        
        datadir+'nm_homicide-rate.csv', 
        datadir+'nm_GEDEvent_v22_1_processed.csv'
        ]


# check
if len(colNames) != len(fileNames) :
    print("ERROR: len(colNames) ",  len(colNames), "must be equal len(fileNames) ", len(fileNames));
    sys.exit();


dfNames=[]
var_min=[]
var_max=[]

var_min0=[]
var_max0=[]

dfNames0=[] # original data (without log normalisation)

k_list = []

# Read WBank files and 
# Create array of dataframes dfNames[i](Code Year colName), a data frame per a property
for i in range( len(colNames) ):    
    colName=colNames[i]
    fileName=fileNames[i]
    print("colName: ", colName)
    dfNames.append( pd.read_csv(fileName, sep=',', header=0) )

    # refine
    # drop rows missing coutry code
    dfNames[i] = dfNames[i].dropna(subset=['Code'])
    if(i > 0 ):
        dfNames[i] = dfNames[i].drop(columns=['Entity'])
    print( dfNames[i][colName].dtypes)    

    # save original
    dfNames0.append( dfNames[i].copy(deep=True) );

    # find min, max aof original data
    var_min0.append( dfNames[i][colName].min() )
    var_max0.append( dfNames[i][colName].max() )


    # log transform positive vars 
    # NMY
    #k_list.append( 1. / (dfNames[i][colName].median() + 1.e-9) );
    k_list.append( 10.0 );
    print(colName, k_list[i]);        
    k = k_list[i];

    if len(logaYes) > 0:       
        if colName in logaYes:
            if TRANSF == "LOGA":
                dfNames[i][colName] = np.log(dfNames[i][colName] + 1e-9);
            elif TRANSF == "LOGI":
                Lm = 2.
                L = Lm * var_max0[i];
                dfNames[i][colName] = np.log( dfNames[i][colName] / (L - dfNames[i][colName]) + 1e-5) / k ;
            elif TRANSF == "SQRT":
                dfNames[i][colName] =  np.sqrt( k * abs(dfNames[i][colName]) + 1.e-5 )                
            elif TRANSF == "INVE":
                dfNames[i][colName] = k * dfNames[i][colName] - 1. / ( k * dfNames[i][colName] + 1.e-2 )

        print( dfNames[i][colName])

    # find min, max and normalise all 
    var_min.append( dfNames[i][colName].min() )
    var_max.append( dfNames[i][colName].max() )
    dfNames[i][colName] = (dfNames[i][colName] - var_min[i])/(var_max[i] - var_min[i])

    print(colName, dfNames[i].shape)
    print(colName, dfNames[i].head())

#i = len(dfNames)-1
#dfNames[i]['BattleDeath'] = (1000 * dfNames[i]['BattleDeath']) / dfNames[0]['Population']

#print(len(var_min), len(colNames))

df_min = pd.DataFrame([var_min], columns=colNames)
df_max = pd.DataFrame([var_max], columns=colNames)
df_min.to_csv("df_min.csv");
df_max.to_csv("df_max.csv");


df_min0 = pd.DataFrame([var_min0], columns=colNames)
df_max0 = pd.DataFrame([var_max0], columns=colNames)
df_min0.to_csv("df_min0.csv");
df_max0.to_csv("df_max0.csv");


with open("logaNot.csv", "w") as myfile:
            myfile.write(', '.join(logaNot))

with open("logaYes.csv", "w") as myfile:
            myfile.write(', '.join(logaYes))
#NMY
with open("k_list.csv", "w") as myfile:
            myfile.write(', '.join(str(item) for item in k_list))

with open("transform.csv", "w") as myfile:
            myfile.write(TRANSF)

#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# Combine dataframes from an array "dfNames" into one big dataframe "result"
dfData=dfNames[0].copy(deep=True);
for i in range(1, len(colNames) ):
    dfData = pd.merge(dfData, dfNames[i], how="left", on=["Code", "Year"])
    #dfData = dfData.dropna(subset=['ForestArea'])

# subsample all years above the YEAR_START
dfData = dfData[dfData['Year'] >= YEAR_START]
dfData = dfData[dfData['Year'] <= YEAR_STOP]

#dfData_  = ( dfData.loc[dfData['Year'] >= YEAR_START] ).copy()
#dfData__ = ( dfData_.loc[dfData_['Year'] <= YEAR_STOP]  ).copy()
#dfData = dfData__.copy()

#print(dfData_2017.head())
# replace empty cells with column-mean or median
for c in [c for c in dfData.columns if dfData[c].dtype in numerics and c != 'Year']:
    #x = dfData[c].mean()
    #x = dfData[c].median()
    #dfData[c].fillna(x, inplace = True)
    dfData[c].fillna(method="bfill", inplace = True)


# Same for the original dataframe
# Combine dataframes from an array "dfNames" into one big dataframe "result"
dfData0=dfNames0[0].copy(deep=True);
for i in range(1, len(colNames) ):
    dfData0 = pd.merge(dfData0, dfNames0[i], how="left", on=["Code", "Year"])
# subsample all years above the YEAR_START
dfData0 = dfData0[dfData0['Year'] > YEAR_START]
#print(dfData_2017.head())
# replace empty cells with column-mean or median
for c in [c for c in dfData0.columns if dfData0[c].dtype in numerics and c != 'Year']:
    #x = dfData[c].mean()
    x = dfData0[c].median()
    dfData0[c].fillna(x, inplace = True)


print("result ", dfData.shape)
print(dfData.head())
print(dfData.tail())

# dump to disk
dfData.to_csv("dfData.csv");
dfData0.to_csv("dfData0.csv");


#sys.exit()

# get rid of columns not needed to calculate covariance
dfData_cut = dfData.drop(columns=['Entity','Code','Year']);
print("dfData_cut")
print(dfData_cut.head())


# +++++++++++++++++ Covariance matrix (across all time ansd space)
#cova = dfData_cut.cov();
#cova.to_csv("cova.csv", float_format="%.5f");

# tmp++++++++++++++++++++++ Covariance simulated for every year separately and then integrated across all years to produce mean covariance
dfCov = pd.DataFrame(np.zeros([len(colNames), len(colNames)], dtype=float), index=colNames, columns=colNames)
for year in range(YEAR_START, YEAR_STOP):
    # subsample a year
    dfData_year = ( dfData.loc[dfData['Year']==year] ).copy()
    # drop columns not needed to calculate covariance
    dfData_year_cut = ( dfData.drop(columns=['Entity','Code','Year']) ).copy();
    # get cova_year
    cova_year = dfData_year_cut.cov();    
    dfCov     = dfCov.add(cova_year, fill_value=0)


#print("dfCov before:", dfCov.head())
dfCov = dfCov.div(YEAR_STOP - YEAR_START);    
#print("dfCov after:", dfCov.head())

dfCov.to_csv("cova.csv", float_format="%.5f");

cova = dfCov.copy()
# +++++++++++++++++++++++++

print("dfCov.head()")
print(dfCov.head())


#cova.plot()
#plt.show()

#dfData_cut.plot()
#plt.show()


# get a bunch of correlations
dfData_rab = dfData_cut.copy()
dom_id = 0;
for dom in domains:
    titles_dom = [];
    print(dom)
    # collect all titles for a domain
    for title in colNames:
        items = title.split('_')
        if( int(items[0]) == dom_id ):
            titles_dom.append(title)
    # save mean
    #print(dfData_cut.head())
    #print("columns:" , dfData_cut.columns)
    dfData_rab[dom] = dfData_rab[titles_dom].mean(axis=1);
    dom_id +=1

dfData_dom =  dfData_rab[domains].copy()
dfData_dom.to_csv("dfData_dom.csv");
print("dfData_dom")
print(dfData_dom.head())

# Correlation matrix
corr = dfData_cut.corr();
corr_dom = dfData_dom.corr();

# correlations integrated over domains
corr_rab = corr.copy()
dom_id = 0;
for dom in domains:
    titles_dom = [];
    # collect all titles for a domain
    for title in colNames:
        items = title.split('_')
        if( int(items[0]) == dom_id ):
            titles_dom.append(title)
    # save mean   
    corr_rab[dom] = corr[titles_dom].mean(axis=1);
    dom_id +=1

corr_rab = corr_rab[domains]
corr_rab=corr_rab.transpose()
corr_rab2 = corr_rab.copy()

dom_id = 0;
for dom in domains:
    titles_dom = [];
    # collect all titles for a domain
    for title in colNames:
        items = title.split('_')
        if( int(items[0]) == dom_id ):
            titles_dom.append(title)
    # save mean
    #print("columns:" , dfData_cut.columns)
    corr_rab2[dom] = corr_rab[titles_dom].mean(axis=1);
    dom_id +=1

corr_dom2 = corr_rab2[domains]


print("corr_dom2")
print(corr_dom2.head())


# Covariance matrix
cova_dom = dfData_dom.cov();
cova_dom.to_csv("cova_dom.csv", float_format="%.5f");
#cova_dom2.to_csv("cova_dom2.csv", float_format="%.5f");

fig = plt.figure()
#creating a subplot
ax1 = fig.add_subplot(1,3,1)
f = plt.figure(figsize=(9, 5))
plt.matshow(corr, fignum=f.number, cmap='jet')
plt.xticks(range(dfData_cut.select_dtypes(['number']).shape[1]), dfData_cut.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(dfData_cut.select_dtypes(['number']).shape[1]), dfData_cut.select_dtypes(['number']).columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

ax2 = fig.add_subplot(1,3,2)
f = plt.figure(figsize=(9, 5))
plt.matshow(cova, fignum=f.number, cmap='jet')
plt.xticks(range(dfData_cut.select_dtypes(['number']).shape[1]), dfData_cut.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(dfData_cut.select_dtypes(['number']).shape[1]), dfData_cut.select_dtypes(['number']).columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Covariance Matrix', fontsize=16);

ax3 = fig.add_subplot(1,3,3)
f = plt.figure(figsize=(9, 5))
plt.matshow(corr_dom, fignum=f.number, cmap='jet')
plt.xticks(range(dfData_dom.select_dtypes(['number']).shape[1]), dfData_dom.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(dfData_dom.select_dtypes(['number']).shape[1]), dfData_dom.select_dtypes(['number']).columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);



#plt.matshow(correl)
plt.show()


sys.exit();


