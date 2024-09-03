# Purpose: calculate EnOI (based on Kalman Filter) for multivariate data
# Input: cavarince matrix (cova.csv) and observation data (dfData.csv) produced by "wbank_cov.py"
# Note that data in dfData.csv are normalised to vary from 0 to 1 
# dfData0.csv holds original, NOT normalised data
# Output: ensemble of simulated data
#         fileSim.csv_fine, fileObs.csv,      (NOT normalised aggregated variables integrated over the globe)
#         fileSim.csv_fine, fileObs.csv,      (NOT normalised state variables integrated over the globe)
#         fileSimX.csv_fine, fileSimX.csv, fileObsX.csv_fine, fileObsX.csv  (same as above but for normalised variables)
#         df_sum.csv - summary stats for each country separately
# Author: Nugzar Margvelashvili
# Created: May 2023
# Last updated: 21 August 2024


#importing libraries
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import pandas as pd
import numpy as np
#from mpl_toolkits.basemap import Basemap
import os
os.environ['USE_PYGEOS'] = '0'

#import geopandas
#import folium
import sys;
import math;
#import itertools;

import random
np.random.seed(1)

#https://www.learnpythonwithrune.org/plot-world-data-to-map-using-python-in-3-easy-steps/

# Default vals:
R_MAX = 1.0
refState = []
keyVars_i = []
deltasz = []


# The ordering of vailables must be consistent with the header of "dfData.csv" produced by "wbank_cov.py"
deli = +0
BATTLE_i = +31 + deli
HOMICIDE_i = +30 + deli

RICHSHARE_i = +29 + deli
POPGROWTH_i = +28 + deli
HDI_i = +27 + deli
LIBDEM_i = +26 + deli
LIFEEXP_i = +25 + deli
SCHOOL_i = +24 + deli

RENEWABLES_i = +23 + deli
SCIMIL_i = +22 + deli
SCIGDP_i = +21 + deli

ENERGYPC_i = +20 + deli
MILUSD_i = +19 + deli
GDPPCGROW_i = +18 + deli
GDPPC_i = +17 + deli
FDIOUT_i =  +16 + deli
TRDGDP_i = +15 + deli
POPDENS_i = +14 + deli
POP_i = +13 + deli
GDPGROW_i = +12 + deli
GDP_i = +11 + deli

CEREAL_i = +10 + deli
AGRILAND_i = +9 + deli
FOREST_i =  +8 + deli
LANDA_i = +7 + deli

FERTI_i = +6 + deli
PESTI_i = +5 + deli
POTASH_i = +4 + deli
PHOS_i = +3 + deli
NITRO_i = +2 + deli
CO2_i = +1 + deli
TEMPA_i = 0 + deli

# all vars
allVars = [TEMPA_i, CO2_i, NITRO_i, PHOS_i, POTASH_i, PESTI_i, FERTI_i, 
        LANDA_i, FOREST_i, AGRILAND_i, CEREAL_i, GDP_i, GDPGROW_i, POP_i, POPDENS_i, TRDGDP_i, FDIOUT_i, GDPPC_i,
        GDPPCGROW_i, MILUSD_i, ENERGYPC_i, SCIGDP_i, SCIMIL_i, RENEWABLES_i, SCHOOL_i, LIFEEXP_i, LIBDEM_i, HDI_i, POPGROWTH_i,
        RICHSHARE_i, HOMICIDE_i, BATTLE_i]


#################################################################
################################################################# USER INPUT
# Uncomment one of scenarios below

#'''
# For each country assimilate indicators of this country into its own model (except contaminants, which are assumed to have no observed data).
# The main objective of this simulation is to evaluate the quality of the model by predicting "unobserved"  contaminants.
# According to this scenario, the quality of the model is good when evaluating globally-mean indicators. 
# For a particular country, the quality of predictions varies from one coutry to another (you can simulate specific countries by specifying their codes in stateCodes array below). 
# THe results are not bad for USA nad CHN, so-so for AUS. The quality of the model is poor for GBR, DEU, FRA, unless the global covariance matrix is 
# substituted by the covariance matrix derived from EU coutries only (the covariance is simulated by wbank_cov.py).
SCNR = "DASBASE"
YEAR_START = 1991
YEAR_STOP  = 2017 
R_MAX = 0.50    # obs error scaling factor
#'''

'''
# For each country "aState" assimilate indicators of the reference state (refState) into the model of "aState"
# In other words, indicators of every state are nudged towards the reference state (except land area, and extensive variables population, and gdp)
SCNR = "DAS2C"
YEAR_START = 1991
YEAR_STOP  = 2017
R_MAX = 0.50   # obs error scaling factor (required by DAS2C and DASBASE scenarios only)
refState = ['FJI']  # reference state providing obs for other states to assimilate
'''

'''
# improvement of randomly selected indicators, one indicator per year for every country separately
# a remarkably good agreement with obs; 
SCNR = "PTBMANY"
YEAR_START = 1991
YEAR_STOP  = 2050
keyVars_i = [] 
deltasz = []
'''

'''
# perturbation of selected indicators, every year for every country separately
SCNR = "PTBMANY"
YEAR_START = 1991
YEAR_STOP  = 2050
keyVars_i = [CO2_i, PESTI_i, FERTI_i] # indicators to perturb
deltasz = [-0.005, -0.005, -0.005]    # annual perturbation values (normalised space, indicators ranging from 0 to 1)
'''


# Select countries to simulate
# stateCodes has a priority over the selection of the continent
# when stateCodes is NOT empty, countries listed in stateCodes are simulated
# when stateCodes is empty, countries of the CONTINENT are simulated
# when both stateCodes and Continent are empty, the whole world is simulated
# CONTINENT: Africa, Europe, SEAsia, MEast, AmericaN, AmericaS, ALL 
stateCodes = ["FRA"]
CONTINENT = "ALL"


ENS_SIZE = 123;    # Ensemble size

# coeficient to scale errors added to ensemble members
# the errors are sampled from a gaussian with zero mean and the model error covariance
gaussErrorScale = 0.1;


################################################################
################################################################ end USER INPUT

#refState = ['ETH']
#refState = ['TGO']
#refState = ['MLI']
#refState = ['PNG']
###refState = ['NIC']
#refState = ['FJI']
####refState = ['KHM']
#####refState = ['BTN']
#####refState = ['LKA']
#####refState = ['MDG']
####refState = ['IDN']
#refState = ['EST']
#refState = ['GEO']
#refState = ['PHL']
####refState = ['VNM']
#refState = ['EGY']
#refState = ['SWE']
#refState = ['AUS']
#refState = ['FKT']

VERBOSE = 1
domains=["pollutant", "agriculture", "economy", "technology", "social", "conflict"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

##############################
# delogarithmesise
def deloga(Xc, titles, logaYes):
    for i in range(COLS):
        if titles[i] in logaYes:
            Xc[i] = math.exp( max (min(Xc[i], 100), -100)  );
    return Xc;

##############################
# delogit
def delogi(Xc, titles, logaYes, k_list):
    for i in range(COLS):
        if titles[i] in logaYes:
            k = float(k_list[i])
            Lm = 2.
            L = Lm * np_max0[i];
            Xc[i] = L / (1. + math.exp(-1 * max(min(k*Xc[i], 100), -100)) ) ;
    return Xc;


###############################
# desquarise
def deinve(Xc, titles, logaYes, k_list):
    # NMY
    for i in range(COLS):
        k = float(k_list[i])
        if titles[i] in logaYes:
            if abs(Xc[i]) > 10:
                rab =  Xc[i] + abs(Xc[i]);
            else:
                rab =  Xc[i] + math.sqrt(Xc[i] * Xc[i]  + 4.0);
            Xc[i] =  rab  / (2.0*k + 1e-2)
        #print(ix, k, Xc[ix])
    return Xc;

############################

##############################
# delogarithmesise
def desqrt(Xc, titles, logaYes, k_list):
    for i in range(COLS):
        if titles[i] in logaYes:
            k = float(k_list[i])
            Xc[i] = Xc[i] * Xc[i] / k;
    return Xc;
###############################


# subsample dfData for one year
def dfDataPerYear(dfData, year):
    #print("up here");
    df_year = (dfData.loc[dfData['Year']==year]).copy();
    #sys.exit()
    ## get rid of blanks 
    #df_year.loc[:,'Entity'] = df_year['Entity'].replace(" ","");

    df_col = (df_year.Entity.replace(' ', '',regex = True) ).copy()
    df_year['Entity'] = df_col.copy()

    return df_year;

############################
# subsample dfData for a lisit of countries (leave this list of codes empty  to have all countries)
def dfDataStateCode(dfData, codes):
    #df_code = dfData[dfData['Code']==code];
    #print("up here", dfData.head());
    # select subset of rows and all columns
    df_codes = (dfData.loc[dfData['Code'].isin(codes), :]).copy();
    #df_codes_ = df_codes.copy()
    #sys.exit()
    #print("up tere", df_codes.head());
    ## Country names fill blanks with 0
    #df_codes.loc[:,'Entity'] = df_codes['Entity'].replace(" ","") ;


    """
    # Below are some quick examples.
    # Replace Blank values with DataFrame.replace() methods.
    df2 = df.replace(r'^\s*$', np.nan, regex=True)

    # Using DataFrame.mask() method.
    df2=df.mask(df == '')

    # Replace on single column
    df2 = df.Courses.replace('',np.nan,regex = True)

    # Replace on all selected columns
    df2 = df[['Courses','Duration']].apply(lambda x: x.str.strip()).replace('', np.nan)
    """

    df_col = (df_codes.Entity.replace(' ', '',regex = True) ).copy()
    df_codes['Entity'] = df_col.copy()

    #df_codes_  =  (df_codes['Entity'].replace(" ","")).copy();
    #print("up zere", df_codes.head());

    #sys.exit()

    return df_codes;

#########
# transform an additive perturbation (delta) in the original unnormalised space to 
# additive perturbation in a log-normalised space (deltaz)
def delta2deltaz(delta, keyVar_i, keyVar_v, np_min, np_max):
    k = +1e-5;
    k = float(k_list[keyVar_i])
    if keyVar_i in logaYes:
        if TRANSF == "LOGA":
            #log transform
            #gamma = 1.0 + delta / (keyVar_v + 0.001);
            gamma = 1.0 + delta * math.exp(-keyVar_v);
            gammaLog = math.log(gamma)
        elif TRANSF == "LOGI":
            print("LOGI pertirbation scenaio NOT IMPLEMENTED!!!")
            sys.exit()
        elif TRANSF == "INVE":    
            # NMY
            print("INVE pertirbation scenaio NOT IMPLEMENTED!!!")
            sys.exit()
            gammaLog = k * delta + 1./(k*keyVar_v + 1e-2) - 1./(k*(keyVar_v + delta) +1e-2);
        elif TRANSF == "SQRT":         
            gammaLog = math.sqrt(k*keyVar_v*k*keyVar_v + delta) - k*keyVar_v;
    else:
        gammaLog = delta;
    # normalise CPg into CPgn
    # CPgn = ( CPg - CPg_min ) / (CPg_max - CPg_min ) =
    #      = ( logC + log(gamma) - CPg_min ) / (CPg_max - CPg_min) =
    #      = ( logC log(gamma) / (CPg_max - CPg_min) - CPg_min ) / (CPg_max - CPg_min) + log(gamma) / (CPg_max - CPg_min) =
    #      =   Cgn + log(gamma) / (CPg_max - CPg_min)
    # here Cgn is a log normalised value of the unperturbed C:  Cgn = ( logC - Cg_min ) / (Cg_max - Cg_min)
    # and perturbation to log-normalised value is: deltaz =  log(gamma) / (CPg_max - CPg_min)
    # note also in a log transformed space both perturbed (CP) and unperturbed (C) variables
    # have the same min, max values:  CPg_min = Cg_min and CPg_max = Cg_max)
    deltaz = gammaLog / (np_max[keyVar_i] - np_min[keyVar_i])
    return deltaz;

########## same as above except for the array of deltas
def deltas2deltasz(deltas, keyVars_i, keyVars_v, np_min, np_max):

    gammas = 1.0 + deltas / (keyVars_v + 0.001);
    # convert gamma to addditive perturbation of the log transformed value of CP
    # CPg = log(CP) = log( C * gamma) = log(C) + log(gamma)
    gammasLog = np.log(gammas)
    # normalise CPg into CPgn
    # CPgn = ( CPg - CPg_min ) / (CPg_max - CPg_min ) =
    #      = ( logC + log(gamma) - CPg_min ) / (CPg_max - CPg_min) =
    #      = ( logC log(gamma) / (CPg_max - CPg_min) - CPg_min ) / (CPg_max - CPg_min) + log(gamma) / (CPg_max - CPg_min) =
    #      =   Cgn + log(gamma) / (CPg_max - CPg_min)
    # here Cgn is a log normalised value of the unperturbed C:  Cgn = ( logC - Cg_min ) / (Cg_max - Cg_min)
    # and perturbation to log-normalised value is: deltaz =  log(gamma) / (CPg_max - CPg_min)
    # note also in a log transformed space both perturbed (CP) and unperturbed (C) variables
    # have the same min, max values:  CPg_min = Cg_min and CPg_max = Cg_max)
    deltasz = np.zeros(deltas.size)
    j=0;
    for key in keyVars_i:
        deltasz[j] =  gammasLog[j] / (np_max[key] - np_min[key]) 
        j = j + 1;

    #print(type(deltasz))
    #print(deltasz)
    #sys.exit()

    return deltasz;



##############################
def denorm(np_out, np_min, np_max, logaYes, goexp):
    # now thath all countries have been updated
    # get back from log to primitive vars +++++++++++++++++++
    row_index2 = 0;
    for row in np_init:
        Xc = row.copy();
        # denormalise            
        Xc = Xc * (np_max - np_min) + np_min      
        if goexp > 0:
            # delogarithmesise
            for ix in range(COLS):
                if titles[ix] in logaYes:
                    Xc[ix] = math.exp( max (min(Xc[ix], 100), -100)  );
        
        np_out[row_index2] = Xc
        row_index2 = row_index2 + 1        
    return np_out
######################################

def denorm2(np_init, np_min, np_max, logaYes, goexp):
    # now that all countries have been updated
    # get back from log to primitive vars +++++++++++++++++++
    np_out=np.copy(np_init)
    row_index2 = 0;
    for row in np_init:
        Xc = row.copy();
        # denormalise
        Xc = Xc * (np_max - np_min) + np_min
        if goexp > 0 and TRANSF == "LOGA":
            Xc2 = deloga(Xc, titles, logaYes);
        elif goexp > 0 and TRANSF == "LOGI":
            Xc2 = delogi(Xc, titles, logaYes, k_list);
        elif goexp > 0 and TRANSF == "SQRT":
            Xc2 = desqrt(Xc, titles, logaYes, k_list);   
        elif goexp > 0 and TRANSF == "INVE":
            Xc2 = deinve(Xc, titles, logaYes, k_list);            
        else:
            Xc2 = Xc.copy()

        np_out[row_index2] = Xc2.copy()
        row_index2 = row_index2 + 1
        #sys.exit()

    return np_out
######################################




def scoring(ens_count, year, df_sum, np_out, domains, domIDsi, fileOut):

    # score    
    #for dom in domains:
    #    df_sum[dom]=np.nan;
    #df_sum['all']=np.nan;
    #print(df_sum.head())

    row_index3 = 0
    glob_scores_agre=np.zeros(len(domains)+1)
    glob_scores_fine = np.zeros(COLS)
    #glob_all =0;
    # loop over countries
    for row in np_out:
        Xa = row
        # score
        agre_scores=np.zeros(len(domains)+1)
        fine_scores=np.zeros(COLS)
        #r_all = 0.0;
        lenXa = len(Xa); 
        # for a given country, loop over domains
        # and calculate scores
        for j in range(len(domains)):
            k=0;
            for i in range(lenXa):
                if int(domIDs[i]) == j:
                    agre_scores[j] += float(Xa[i]) * float(scales[i]);
                    k += 1;
            # accumulate total        
            agre_scores[-1] += agre_scores[j]/lenXa;       
            if int(k)>0:
                agre_scores[j] =  agre_scores[j] / k;

        k=0;
        # global score for individual indicators
        for j in range(COLS):
            fine_scores[j] += float(Xa[j]) * float(scales[j]);
            k += 1;
        if int(k)>0:
            fine_scores[j] =  fine_scores[j] / k;

        # for a given ensemble, year, country, write scores to df_sum
        if fileOut == "fileSimX.csv":
            #if( int(year) == (int(YEAR_STOP)-1) ):
            #print(df_sum.head())
            rab0 = float(ENS_SIZE) * (float(YEAR_STOP) - float(YEAR_START))
            for j in range(len(domains)):
                rab = df_sum.at[row_index3, domains[j]]
                df_sum.at[row_index3, domains[j]] =  float(rab) + agre_scores[j] / rab0;   
            rab = df_sum.at[row_index3, 'all']    
            df_sum.at[row_index3, 'all'] =  float(rab) + agre_scores[-1] / rab0;



        #print(row_index3, r_envi, r_econ, r_soci, r_mean)

        row_index3 = row_index3 + 1
        
        # for a given emnsemble, year, country, add this coutry's score to the global score accumulator
        # (this global score is reset to 0 every new year)
        i=0;
        for j in range(len(domains)+1):
            glob_scores_agre[j] += agre_scores[j]
        #glob_all += r_all

        for j in range(COLS):
            glob_scores_fine[j] += fine_scores[j]
        
        # completed loop over the countries
    
    ########### for a given ensemble-member and given year WRITE TO FILES
    if fileOut == "fileSimX.csv":
        if(  int(year) == (int(YEAR_STOP)-1) and  ens_count == (int(ENS_SIZE) - 1) ):
            df_sum.to_csv("df_sum.csv", mode='w', header="true");
            #else:
            #    df_sum.to_csv("df_sum.csv", mode='a', header="false");
            # reset to 0, so that for the next year it calculates a new mean value   
            for col in domains:
                df_sum[col] = 0.0
            df_sum['all'] = 0.0


    # for a given ensemble and year
    # append global domain-aggregated scores to the output file
    if( int(year) == YEAR_START  and ens_count == 0):
        with open(fileOut, "w") as myfile:
            myfile.write( "Year ")
            for j in range(len(domains)):
                myfile.write( domains[j] + " ")                
            myfile.write( "all \n" )

    with open(fileOut, "a") as myfile:
        myfile.write( str(year) + " ")
        for j in range(len(domains)+1):
            myfile.write( np.array2string(glob_scores_agre[j]/row_index3) + " ")
        myfile.write( "\n" )    

    if( int(year) == YEAR_STOP - 1 ):
        print(ens_count)
        with open(fileOut, "a") as myfile:
            myfile.write( "\n" )



    # for a given ensemble and year
    # append global individual-scores to the output file
    fileOut_fine = fileOut+"_fine"
    if( int(year) == YEAR_START  and ens_count == 0):
        with open(fileOut_fine, "w") as myfile:
            myfile.write( "Year ")
            for j in range(COLS):
                myfile.write( titles[j] + " ")
            myfile.write( "all \n" )

    with open(fileOut_fine, "a") as myfile:
        myfile.write( str(year) + " ")
        for j in range(COLS):
            myfile.write( np.array2string(glob_scores_fine[j]/row_index3) + " ")
        myfile.write( "\n" )

    if( int(year) == YEAR_STOP - 1 ):
        print(ens_count)
        with open(fileOut_fine, "a") as myfile:
            myfile.write( "\n" )



##############################

# generate n samples from the gaussian with 
# mean m[d], and covariance K_0[d, d]
def sampleGauss(n, d, m, K_0):
    # Define epsilon.
    #epsilon = 1e-4
    epsilon =  1e-3
    #epsilon =  1e-2
    # Add small pertturbation.
    K = K_0 + epsilon*np.identity(d)
    z = np.random.multivariate_normal(mean=m.reshape(d,), cov=K, size=n)
    #y = np.transpose(z)
    #print("np.shape(y)=", np.shape(y))
    return z

###############################################


# continent: Africa, Europe, SEAsia, MEast, AmericaN, AmericaS, ALL
# return list of state codes to simulate over
def continent_(CONTINENT):

    stateCodes0 = dfData['Code'].unique().tolist()
    #stateCodes = stateCodes[100:]

    #continent = "ALL"

    if CONTINENT == "Africa":
        # Africa
        stateCodes2 = ['DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CPV', 'CAF', 'TCD', 'COM', 'COD', 'DJI', 'EGY', 'GNQ', 'ERI', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB', 'CIV', 'KEN', 'LSO', 'LBR', 'LBY', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MYT', 'MAR', 'MOZ', 'NAM', 'NER', 'NGA', 'COG', 'REU', 'RWA', 'SHN', 'STP', 'SEN', 'SYC', 'SLE', 'SOM', 'ZAF', 'SDN', 'SWZ', 'TZA', 'TGO', 'TUN', 'UGA', 'ESH', 'ZMB', 'ZWE']
        # stateCodes2 = [ 'SSD'] bad apple
        
    elif CONTINENT == "Europe":
        # Europe
        stateCodes2 = ['ALA', 'ALB', 'AND', 'AUT', 'BLR', 'BEL', 'BIH', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FRO', 'FIN', 'FRA', 'DEU', 'GIB', 'GRC', 'GGY', 'HUN', 'ISL', 'IRL', 'IMN', 'ITA', 'JEY', 'XKX', 'LVA', 'LIE', 'LTU', 'LUX', 'MKD', 'MLT', 'MDA', 'MCO', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SMR', 'SRB', 'SCG', 'SVK', 'SVN', 'ESP', 'SJM', 'SWE', 'CHE', 'UKR', 'GBR', 'VAT'] 
    elif CONTINENT == "Asia":    
        # Asia
        stateCodes2 =['AFG', 'ARM', 'AZE', 'BHR', 'BGD', 'BTN', 'IOT', 'BRN', 'KHM', 'CHN', 'CCK', 'GEO', 'HKG', 'IND', 'IDN', 'IRN', 'ISR', 'JPN', 'JOR', 'KAZ', 'KWT', 'KGZ', 'LAO', 'LBN', 'MAC', 'MYS', 'MDV', 'MNG', 'MMR', 'NPL', 'PRK', 'OMN', 'PAK', 'PSE', 'PHL', 'QAT', 'SAU', 'SGP', 'KOR', 'LKA', 'SYR', 'TWN', 'THA', 'TUR', 'TKM', 'ARE', 'UZB', 'VNM', 'YEM']
        #stateCodes2 =['IRQ', 'TJK'] # bad apples
    elif CONTINENT == "SEAsia":
        # South-East Asia
        stateCodes2 =[ 'BHR', 'BGD', 'BTN', 'IOT', 'BRN', 'KHM', 'CHN', 'CCK', 'HKG', 'IND', 'IDN', 'JPN', 'LAO', 'MAC', 'MYS', 'MDV', 'MNG', 'MMR', 'NPL', 'PRK', 'OMN', 'PHL', 'SGP', 'KOR', 'LKA', 'TWN', 'THA', 'VNM']

    elif CONTINENT == "MEast":
        # Middle East
        #stateCodes2 =['IRQ', 'TJK'] # bad apples
        stateCodes2 =['AFG', 'ARM', 'AZE', 'GEO', 'IRN', 'ISR', 'JOR', 'KAZ', 'KWT', 'KGZ', 'LBN', 'OMN', 'PAK', 'PSE', 'QAT', 'SAU', 'SYR', 'TUR', 'TKM', 'ARE', 'UZB', 'YEM']
            
    elif CONTINENT == "Oceania":
        # Oceania
        stateCodes2 =['ASM', 'AUS', 'CXR', 'COK', 'FJI', 'PYF', 'GUM', 'KIR', 'MHL', 'FSM', 'NRU', 'NCL', 'NZL', 'NIU', 'NFK', 'MNP', 'PLW', 'PNG', 'PCN', 'WSM', 'SLB', 'TLS', 'TKL', 'TON', 'TUV', 'VUT', 'WLF'] 
    elif CONTINENT == "AmericaN":
        # America North
        stateCodes2 =['AIA', 'ATG', 'ABW', 'BRB', 'BLZ', 'BMU', 'BES', 'VGB', 'CAN', 'CYM', 'CRI', 'CUB', 'CUW', 'DMA', 'DOM', 'SLV', 'GRL', 'GRD', 'GLP', 'GTM', 'HTI', 'HND', 'JAM', 'MTQ', 'MEX', 'MSR', 'ANT', 'NIC', 'PAN', 'PRI', 'BLM', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT', 'SXM', 'BHS', 'TTO', 'TCA', 'USA', 'VIR'] 
    elif CONTINENT == "AmericaS":
        # America South
        # stateCodes2 =['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'FLK', 'GUF', 'GUY', 'PRY', 'PER', 'SUR', 'URY', 'VEN']
        stateCodes2 =['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'PRY', 'PER', 'SUR', 'URY', 'VEN']
        #stateCodes2 =['GUY','FLK', 'GUF'] # bad apples
    else:
        stateCodes0.remove('IRQ')
        stateCodes0.remove('TJK')
        stateCodes0.remove('GUY')
        stateCodes0.remove('SSD')

        #stateCodes0.remove('FLK')
        #stateCodes0.remove('GUF')
        stateCodes2 = stateCodes0;


    stateCodes = []
    for item in stateCodes2:
        if item in stateCodes0:
            stateCodes.append(item);

    return stateCodes;




################################# #######################################################
################################ ########################################################           MAIN

file_states='states.csv'
df_states = pd.read_csv(file_states, sep=',', header=0)
print(df_states.head())

# initial data
# read all data (normalised values in dataFile)
dataFile = "dfData.csv"
dfData = pd.read_csv(dataFile, sep=',', header=0)

# countriess to simulate
if len(stateCodes) < 1:
    stateCodes = continent_(CONTINENT)

print(len(stateCodes))
print(stateCodes)

# select countries
if len(stateCodes) > 0:
    dfData=dfDataStateCode(dfData, stateCodes)

# same for the original (not normalised data)
dataFile0 = "dfData0.csv"
dfDataRaw = pd.read_csv(dataFile0, sep=',', header=0)
# select countries
if len(stateCodes) > 0:
    dfDataRaw=dfDataStateCode(dfDataRaw, stateCodes)

#dfData.to_csv("abefore.csv", float_format="%.5f");

# replace empty cells with column-mean or median
for c in [c for c in dfData.columns if dfData[c].dtype in numerics and c != 'Year']:
    #x = dfData[c].mean()
    #x = dfData[c].median()
    dfData[c].fillna(method="bfill", inplace = True)

for c in [c for c in dfDataRaw.columns if dfDataRaw[c].dtype in numerics and c != 'Year']:
    #x = dfData[c].mean()
    #x = dfDataRaw[c].median()
    dfDataRaw[c].fillna(method="bfill", inplace = True)

#dfData.to_csv("after.csv", float_format="%.5f");

# min max of the original columns (proor to normalisaton)
df_min = pd.read_csv("df_min.csv", sep=',', header=0)
df_max = pd.read_csv("df_max.csv", sep=',', header=0)

df_min0 = pd.read_csv("df_min0.csv", sep=',', header=0)
df_max0 = pd.read_csv("df_max0.csv", sep=',', header=0)


if os.stat("logaYes.csv").st_size != 0:
    df_logaYes =  pd.read_csv("logaYes.csv", sep=',', header=0)
    logaYes = list(df_logaYes.columns)
    logaYes = [ item.strip() for item in logaYes]
else:
    logaYes = [];

#NMY
if os.stat("k_list.csv").st_size != 0:

    with open("k_list.csv", "r") as file1:
        k_list = [float(i) for line in file1 for i in line.split(',') if i.strip()]
    """
    df_k_list =  pd.read_csv("k_list.csv", sep=',', header=0, dtype='float64')
    df_k_list = df_k_list.astype('float64')
    print(df_k_list.head())
    k_list = list(df_k_list.columns)
    k_list = [ item.strip() for item in k_list]
    """ 
else:
    k_list = [];

if os.stat("transform.csv").st_size != 0:
    with open("transform.csv", "r") as file1:
        TRANSF = file1.readline().strip('\n').strip()
        print("item= ", TRANSF)
else:
    TRANSF = "LINE";

print("TRANSF= ", TRANSF) 

print(k_list)

# subsample a year to start
df_init = dfDataPerYear(dfData, YEAR_START);

if VERBOSE == 1:
    print(df_init.head())
    print("df_init.shape()=: ", df_init.shape)

df_sum = df_init.copy()
#df_sumf = df_init.copy()

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

# subsample 3 columns
df_sum = df_sum[['Entity','Year','Code']]
for col in domains:
    df_sum[col] = 0.0
df_sum['all'] = 0.0    
df_sum.reset_index(drop=True, inplace=True)

df_min = df_min.drop(['Unnamed: 0'], axis=1)
np_min = (df_min.to_numpy()).flatten()
df_max = df_max.drop(['Unnamed: 0'], axis=1)
np_max = (df_max.to_numpy()).flatten()

df_min0 = df_min0.drop(['Unnamed: 0'], axis=1)
np_min0 = (df_min0.to_numpy()).flatten()
df_max0 = df_max0.drop(['Unnamed: 0'], axis=1)
np_max0 = (df_max0.to_numpy()).flatten()



np_init0 = np.copy(np_init)

# covariance
file_P='cova.csv'
df_P = pd.read_csv(file_P, sep=',', header=0)
print("df_P.shape: ", df_P.shape)
print(df_P.head())

df_P.drop(df_P.columns[0], axis=1, inplace=True)
#df_P = np.delete(df_P, 0, 1)
ROWS, COLS = df_P.shape
print("df_P.shape: ", ROWS, COLS)
print(df_P.head())

P = df_P.copy()
P = np.array(P,dtype=float)
print(P.dtype)
print("P.shape: ", np.shape(P))


# NMY 2024 tmp increasdd correlations with temp anomaly
# Works well for a global set of countries
# Not so good when running over EU subset
P[:,0] *= 10;
P[0,:] *= 10;
P[0,0] = 1.0;


for ens_count in range(0, ENS_SIZE):
    np_init = np.copy(np_init0)
    np_out  = np.copy(np_init0)
    print("ensemble=: ", ens_count)
    for year in range(YEAR_START, YEAR_STOP + 0):
        #np_out  = np.copy(np_init0)

        # extending obs into the future if needed
        if year <= 2017:
            year_f = year
        else:
            year_f = 2017
        #print("year_f= ",year_f)    
        # forcing / observations data
        dfData_year = dfDataPerYear(dfData, year_f);
        dfData_year_code = pd.merge(dfData_year, codes, how="right", on=["Code"])
        dfData_year = dfData_year_code.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
        dfData_year = dfData_year.replace(np.nan,0)
        npData_year = dfData_year.to_numpy()
        # original, NOT log-normalised values
        dfDataRaw_year = dfDataPerYear(dfDataRaw, year_f);
        dfDataRaw_year_code = pd.merge(dfDataRaw_year, codes, how="right", on=["Code"])
        dfDataRaw_year = dfDataRaw_year_code.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
        dfDataRaw_year = dfDataRaw_year.replace(np.nan,0)
        npDataRaw_year = dfDataRaw_year.to_numpy()


        #print("np_init.shape: ", np.shape(np_init))
        #print("np_Data_year.shape: ", np.shape(npData_year))
        '''
        deli = +0
        BATTLE_i = +31 + deli
        HOMICIDE_i = +30 + deli

        RICHSHARE_i = +29 + deli
        POPGROWTH_i = +28 + deli
        HDI_i = +27 + deli
        LIBDEM_i = +26 + deli
        LIFEEXP_i = +25 + deli
        SCHOOL_i = +24 + deli

        RENEWABLES_i = +23 + deli
        SCIMIL_i = +22 + deli
        SCIGDP_i = +21 + deli 

        ENERGYPC_i = +20 + deli
        MILUSD_i = +19 + deli 
        GDPPCGROW_i = +18 + deli
        GDPPC_i = +17 + deli
        FDIOUT_i =  +16 + deli
        TRDGDP_i = +15 + deli
        POPDENS_i = +14 + deli
        POP_i = +13 + deli
        GDPGROW_i = +12 + deli
        GDP_i = +11 + deli

        CEREAL_i = +10 + deli
        AGRILAND_i = +9 + deli
        FOREST_i =  +8 + deli
        LANDA_i = +7 + deli

        FERTI_i = +6 + deli        
        PESTI_i = +5 + deli
        POTASH_i = +4 + deli
        PHOS_i = +3 + deli
        NITRO_i = +2 + deli
        CO2_i = +1 + deli
        TEMPA_i = 0 + deli
        '''

        row_index = 0
        # loop over countries
        for row in np_init: 
            #print("\n")
            #print(row[1], row[2])
            Xa = row #.values
            
            Xb = Xa.copy()
            #Xc = Xa.copy()

            # Obs
            Y  = np.zeros(COLS)
            # Obs error covariance
            R = R_MAX * np.identity(COLS)
            np.array(R,dtype='float64')
          

            #R[GDP_i, GDP_i]                = 10.30 * R_MAX;
            #R[POP_i, POP_i]                = 10.30 * R_MAX;

            ''' 
            R[RICHSHARE_i, RICHSHARE_i]    = 0.32 * R_MAX;
            #R[FERTI_i, FERTI_i]            = 0.10 * R_MAX;
            #R[TEMPA_i, TEMPA_i]            = 0.10 * R_MAX;
            R[GDPGROW_i, GDPGROW_i]        = 0.31 * R_MAX;
            R[GDPPCGROW_i, GDPPCGROW_i]    = 0.31 * R_MAX;
            R[GDP_i, GDP_i]                = 0.30 * R_MAX;
            R[POP_i, POP_i]                = 0.30 * R_MAX;
            R[POPDENS_i, POPDENS_i]        = 0.30 * R_MAX;
            R[LANDA_i, LANDA_i]            = 0.30 * R_MAX;
            R[AGRILAND_i, AGRILAND_i]      = 0.30 * R_MAX
            R[TRDGDP_i, TRDGDP_i]          = 0.30 * R_MAX
            R[SCIGDP_i, SCIGDP_i]          = 0.32 * R_MAX;
            R[SCIMIL_i, SCIMIL_i]          = 0.32 * R_MAX;
            R[LIFEEXP_i, LIFEEXP_i]        = 0.32 * R_MAX;
            R[SCHOOL_i, SCHOOL_i]          = 0.32 * R_MAX;
            R[ENERGYPC_i, ENERGYPC_i]      = 0.30 * R_MAX;
            '''

            # Measurement mapping matrix
            H = np.zeros((COLS, COLS))
            #print(H.dtype)
            #print("H.shape: ", np.shape(H))

            #deltaYear = year - YEAR_START
            #KD = 0.1                
            #coln=17
            #aim_pcent = 0.85
            #aim = np.quantile(np_init0[:, coln], aim_pcent)
            #rab = np_init0[row_index, coln]
            #Y[coln] =  rab + (aim - rab) * (1 - np.exp(-KD * deltaYear))
            #H[coln][coln] = 1;
            #print(aim, rab)
            
            # all vars
            #allVars = [TEMPA_i, CO2_i, NITRO_i, PHOS_i, POTASH_i, PESTI_i, FERTI_i, 
            #              LANDA_i, FOREST_i, AGRILAND_i, CEREAL_i, GDP_i, GDPGROW_i, POP_i, POPDENS_i, TRDGDP_i, FDIOUT_i, GDPPC_i,
            #              GDPPCGROW_i, MILUSD_i, ENERGYPC_i, SCIGDP_i, SCIMIL_i, RENEWABLES_i, SCHOOL_i, LIFEEXP_i, LIBDEM_i, HDI_i, POPGROWTH_i,
            #              RICHSHARE_i, HOMICIDE_i, BATTLE_i]

            ##############################
            if( SCNR == "DASBASE" ):
                # testing against obs
                # assimilate all except contaminants
                forcedVars = [LANDA_i, FOREST_i, AGRILAND_i, CEREAL_i, GDP_i, GDPGROW_i,  POP_i, POPDENS_i, TRDGDP_i, FDIOUT_i, GDPPC_i,
                          GDPPCGROW_i, MILUSD_i, ENERGYPC_i, SCIGDP_i, SCIMIL_i, RENEWABLES_i, SCHOOL_i, LIFEEXP_i, LIBDEM_i, HDI_i, POPGROWTH_i,
                          RICHSHARE_i, HOMICIDE_i, BATTLE_i]

                # all except contaminants and GDP_i POP_i (to be updated via correlation linkages with the observed variables)
                #forcedVars = [LANDA_i, FOREST_i, AGRILAND_i, CEREAL_i, GDPGROW_i,  POPDENS_i, TRDGDP_i, FDIOUT_i, GDPPC_i,
                #              GDPPCGROW_i, MILUSD_i, ENERGYPC_i, SCIGDP_i, SCIMIL_i, RENEWABLES_i, SCHOOL_i, LIFEEXP_i, LIBDEM_i, HDI_i, POPGROWTH_i,
                #              RICHSHARE_i, HOMICIDE_i, BATTLE_i]

                for coln in forcedVars:
                    Y[coln] = npData_year[row_index, coln];
                    H[coln][coln]=1.

            
                ############################
            if( SCNR == "DAS2C" ):
                    
                # assimilate all except contaminants and extensive variables GDP_i and POP_i (to be updated via ENOI inferrence)
                # Land area is kept constant (see below)
                forcedVars = [LANDA_i, FOREST_i, AGRILAND_i, CEREAL_i, GDPGROW_i,  POPDENS_i, TRDGDP_i, FDIOUT_i, GDPPC_i,
                          GDPPCGROW_i, MILUSD_i, ENERGYPC_i, SCIGDP_i, SCIMIL_i, RENEWABLES_i, SCHOOL_i, LIFEEXP_i, LIBDEM_i, HDI_i, POPGROWTH_i,
                          RICHSHARE_i, HOMICIDE_i, BATTLE_i]

                forcedVars_v = np.array( [ np_init[row_index, key] for key in forcedVars ] )
                #Y[coln] = npData_year[row_index, coln];
                # get current observed values of the keyVars for the selected reference (target) country
                df_state2follow = dfData_year_code[ dfData_year_code['Code'].isin(refState) ];
                df_state2follow = df_state2follow.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
                np_state2follow = df_state2follow.to_numpy()
                refVals =  np.array( [ np_state2follow[0, key] for key in forcedVars] )
                                   
                n=+0;
                for coln in forcedVars:
                    coln = int(coln);
                    Y[coln] = refVals[n];
                    H[coln][coln] = 1;
                    n += 1;

                # keep own land area constant
                # fixed vars [LandArea, Population]
                #fixedVars = [LANDA_i, POP_i, POPDENS_i]
                #fixedVars = [LANDA_i, GDP_i, POP_i]
                fixedVars = [LANDA_i]
                for coln in fixedVars:
                    Y[coln] = npData_year[row_index, coln];
                    H[coln][coln]=1.

                #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")

            if( SCNR == "DASBASE" or SCNR == "DAS2C"):
                # ENOI MATRIX OPERATIONS                
                #print(P.shape, H.shape)

                # W = PHt * Inv(R + HPHt)
                # Xa = Xb + W(Y-HXb)
                W     = np.zeros((COLS, COLS))
                np.array(W,dtype=float)

                W_inv = np.zeros((COLS, COLS))
                np.array(W_inv,dtype=float)

                PHt = np.matmul(P, np.transpose(H))
                np.array(PHt,dtype=float)
                W = R + np.matmul(H, PHt)
                        
                #print(W)
                
                W_inv = np.linalg.inv(W)
                W = np.matmul(PHt, W_inv)
                #print("W: ")
                #print(W)

                Delta = Y - np.matmul(H,Xb)
                Xa = Xb + np.matmul(W, Delta)
                # add noise
                n=1
                dim=COLS
                m=np.zeros(COLS)
                samples = sampleGauss(n, dim, m, P);
                #print("samples=", samples.flatten());
                #if(row_index==1):
                #Xa = Xa + 0.1 * np.random.randn(COLS)
                Xa = Xa + gaussErrorScale*samples.flatten()

            
                # constraints
                """
                scale1= 1.e-6

                LANDA = scale1 * ( Xa[LANDA_i] * (np_max[LANDA_i] - np_min[LANDA_i]) + np_min[LANDA_i] )
                POPDENS   = ( Xa[POPDENS_i] * (np_max[POPDENS_i] - np_min[POPDENS_i]) + np_min[POPDENS_i] )
                POP = LANDA * POPDENS
                Xa[POP_i] = (POP - scale1 * np_min[POP_i]) / (np_max[POP_i] - np_min[POP_i])
                Xa[POP_i] =  Xa[POP_i] / scale1
                """
                """
                POP   = scale1 * ( Xa[POP_i] * (np_max[POP_i] - np_min[POP_i]) + np_min[POP_i] )
                GDPPC = ( Xa[GDPPC_i] * (np_max[GDPPC_i] - np_min[GDPPC_i]) + np_min[GDPPC_i] )
                GDP = GDPPC * POP 
                Xa[GDP_i] = (GDP - scale1 * np_min[GDP_i]) / (np_max[GDP_i] - np_min[GDP_i])
                Xa[GDP_i] =  Xa[GDP_i] / scale1
                """
                """
                POP   = scale1 * ( Xa[POP_i] * (np_max[POP_i] - np_min[POP_i]) + np_min[POP_i] )
                CO2PC = ( Xa[CO2PC_i] * (np_max[CO2PC_i] - np_min[CO2PC_i]) + np_min[CO2PC_i] )
                CO2 = CO2PC * POP
                Xa[CO2_i] = (CO2 - scale1 * np_min[CO2_i]) / (np_max[CO2_i] - np_min[CO2_i])
                Xa[CO2_i] =  Xa[CO2_i] / scale1
                """

                #Xa = np.exp(Xa.astype(float))
                # clip out negative valuesa ++++++++++++++++++++++++++++++
                #for i in range (Xa.shape[0]):
                #    if float(Xa[i]) < 1e-5:
                #        Xa[i] = 1e-5;

                #Xa = np.clip(Xa, 0, None)
                #Xa = np.clip(Xa, 0, 1)
                #print(Xa)
                np_init[row_index] = Xa
        
                #print("XXXXXXXXXXXXXXXXXXXXXXXXXX end DASEBASE and DAS2C)


            if( SCNR == "PTBMANY" ):
                ###########
                #print("year=", year);
                #print("row_index=", row_index);
                Xb  = Xb.flatten();
                # Xb = np.where(Xb<0, 0.9*Xb, Xb); # tmp
                # Xb = np.where(Xb>2, 2, Xb);    # tmp

                scales = np.array(scales)
                scales = scales.astype(np.float64)


                if not keyVars_i: # if list is empty
                    # get keyVar_index to prturb
                    keyVar_i = random.randint(6*0, int(np.shape(Xb)[0])-1-0 ) # best random
                    #keyVar_i = int(GDPPC_i); # baseline
                    #keyVar_i = int(ENERGYPC_i);

                    # get an increment value (keyVar delta)
                    #delta0 = 0;      # idle/freez run
                    delta0 = +0.005; #  (best)
                    #delta0 = +0.01;  # (double)
                    #delta0 = +0.0025; # (half)
                    #delta0 = -0.005; # downgrade

                    # feedback
                    #delta0 = delta0 - 0.01 * math.fabs(Xb[keyVar_i])
                    #delta0 = delta0 - 0.005 * (math.fabs(Xb[GDPPC_i]) + math.fabs(Xb[RICHSHARE_i])) # decadence break
                    #delta0 = delta0 - 0.0025 * (math.fabs(Xb[CO2_i]) + math.fabs(Xb[FERTI_i]) + math.fabs(Xb[PESTI_i]))
                    #delta0 = delta0 - 0.0025 * (math.fabs(Xb[RICHSHARE_i]) + math.fabs(Xb[ HOMICIDE_i]) + math.fabs(Xb[ BATTLE_i]));
                    #delta0 = delta0 - delta0 * (1./6) * (math.fabs(Xb[CO2_i]) + math.fabs(Xb[FERTI_i]) + math.fabs(Xb[PESTI_i]) +
                    #                                     math.fabs(Xb[RICHSHARE_i]) + math.fabs(Xb[ HOMICIDE_i]) + math.fabs(Xb[ BATTLE_i]));
                    # if( float(Xb[keyVar_i]) > 0.2):
                    #     delta = delta0 * float( scales[keyVar_i] );
                    # else:
                    #     delta = delta0 * (math.fabs(Xb[keyVar_i]) + 1e-2) * float( scales[keyVar_i] );

                    #if( int(domIDs[keyVar_i]) == 2):
                    #    delta0 = 0.0; # downgrade/freez economy

                    if( float(scales[keyVar_i]) < 0 ):
                        delta = -delta0;
                    else:
                        delta = +delta0;

                    #delta = 0; # random
                    #delta = delta0; # baseline

                    #allVars_v = np.array( [ np_init[row_index, key] for key in allVars ] )

                    #keyVars_i =  [ GDP_i, TRDGDP_i, MILUSD_i]
                    #rats = [  1.0,     1.0,      1.0 ] ; #

                    #if keyVar_i not in keyVars_i:
                    #    keyVars_i.append(keyVar_i);
                    #    rats.append(1.0);

                    keyVars_i = [keyVar_i]

                    #NkeyVars = len(keyVars_i)
                    #NallVars = len(allVars)
                    #deltasz = np.zeros((NkeyVars,), dtype=float)
                    #deltasz = deltasz.astype(np.float64)

                    deltasz = []
                    deltasz.append(delta)


                ######## EVENTS
                '''
                ######## FIN Crisis
                if( int(year) == 2009 ):
                    keyVars_i = [GDPGROW_i, GDPPCGROW_i, TRDGDP_i]
                    deltasz[0] = -0.002;
                    deltasz.append(-0.01);
                    deltasz.append(-0.005);
                    #keyVars_i = [GDPGROW_i];
                    #deltasz[0] = -0.002;
                if( int(year) == 2010 ):
                    keyVars_i = [GDPGROW_i, GDPPCGROW_i, TRDGDP_i]
                    deltasz[0] = +0.002;
                    deltasz.append(+0.01);
                    deltasz.append(+0.005);
                    #keyVars_i = [GDPGROW_i];
                    #deltasz[0] = +0.002;
                '''

                '''
                ######## COVID
                if( int(year) == 2020 ):
                    keyVars_i = [GDP_i, TRDGDP_i]
                    deltasz[0] = -0.01;
                    deltasz.append(-0.01);
                #if( int(year) == 2021 ):
                #    keyVars_i = [GDP_i, TRDGDP_i]
                #    deltasz[0] = +0.01;
                #    deltasz.append(+0.01);
                '''
                '''
                ######## WAR
                if( int(year) == 2022 ):
                    keyVars_i = [MILUSD_i, HOMICIDE_i, BATTLE_i]
                    deltasz[0] = +0.01;
                    deltasz.append(+0.01);
                    deltasz.append(+0.01);

                '''

                '''
                ######### Fixed CO2
                #if( int(year) > 2024 and keyVar_i == CO2_i):
                #    deltasz[0] = +0.0;

                ######### Increased CO2
                if( int(year) == 2025):
                    keyVars_i = [CO2_i]
                    deltasz[0] = +0.01;
                ################
                '''

                deltasz = np.array(deltasz)
                deltasz = deltasz.astype(np.float64)

                NkeyVars = len(keyVars_i)
                NallVars = len(allVars)


                # handle covariates from the list of forced variables (ie variables designated as being obsevations),
                # so that they (observations) are consistent with the perturbed key variables
                P22 = [ P[i][j] for i in keyVars_i for j in keyVars_i ]
                P22 = np.asarray(P22)
                P22 = P22.reshape(NkeyVars,NkeyVars)
                P22 = np.add(P22, 1.e-7 * np.identity(NkeyVars))
                #print(np.shape(P22))
                #print(P22)
                #P12 = [ P[i][j] for i in keyVars_i for j in allVars ]
                P12 = [ P[i][j] for i in allVars for j in keyVars_i ]
                P12 = np.asarray(P12)
                P12 = P12.reshape(NallVars,NkeyVars)
                #print(np.shape(P12))
                #print(P12)
                #print("P22:")
                #print(P22)
                #sys.exit()

                P22_inv = np.linalg.inv(P22)
                ptb_list = np.matmul(P12, P22_inv)

                ptb_list = list( np.matmul(ptb_list, deltasz) )

                ptb_list  = np.array(ptb_list);
                ptb_list  = ptb_list.flatten();

                #print("PTBMANY")
                #print(ptb_list)

                #print("np.shape(P_shape=", np.shape(P));
                #print("np.shape(ptb_list=", np.shape(ptb_list));
                Xb = Xb + ptb_list;

                # add noise
                #n=int(np.shape(Xb)[0]);
                n=1;
                dim=COLS
                m=np.zeros(COLS)
                samples = sampleGauss(n, dim, m, P);
                #print("samples: ", samples);
                #sys.exit();

                #samples = sampleGauss(n, dim, m, np.identity(dim));
                #print("samples.flatten()=", samples.flatten(), " mean=", np.mean(samples.flatten()));
                Xb = Xb + gaussErrorScale*samples.flatten();

                #Xb = [ float(item) + random.random()-0.5  for item in Xb ]

                # vars fixed to obs
                # Fixed CO2
                #if( int(year) > 2024 ):
                #    fixedVars = [LANDA_i, CO2_i]
                #else:
                #    fixedVars = [LANDA_i]

                fixedVars = [LANDA_i]

                for coln in fixedVars:
                    Xb[coln] = npData_year[row_index, coln];

                # write to init pool
                np_init[row_index] = Xb;

                #print("XXXXXXXXXXXXXXXXXXXXXXXXXX end PTBMANY")

            row_index = row_index + 1
            #print("Year: ", year)
            # end coutry iteration
            ##################################

        
        # now thath all countries have been updated
        # get back from normalised to log vars +++++++++++++++++++        
        goexp = +0
        #np_out=np.copy(np_init)
        #np_out = denorm(np_out, np_min, np_max, logaYes, goexp)
        np_out = denorm2(np_init, np_min, np_max, logaYes, goexp)

        # get scoring for a given year
        scoring(ens_count, year, df_sum, np_out, domains, domIDs, "fileSim.csv");

        # plot log - normalised values
        #goexp = +0
        np_out=np.copy(np_init)
        #denorm(np_out, np_min, np_max, logaYes, goexp)
        # get scoring for a given year
        scoring(ens_count, year, df_sum, np_out, domains, domIDs, "fileSimX.csv");

        # end year iteration

    # end ensemble iterations
    ###############################################################################

print()
# get observations
ens_count=0
for year in range(YEAR_START, YEAR_STOP):
    if year <= 201700:
        year_f = year
    else:
        year_f = 2017
    #print("year_f= ",year_f)
    # forcing / observations data
    dfData_year = dfDataPerYear(dfData, year_f);
    dfData_year = dfData_year.drop(['Unnamed: 0','Entity','Year','Code'], axis=1)
    dfData_year = dfData_year.replace(np.nan,1e-5)
    npData_year = dfData_year.to_numpy()
    
    # get back tp log vars +++++++++++++++++++++++++
    goexp = +0
    #np_out=np.copy(npData_year)
    #np_out = denorm(npData_year, np_min, np_max, logaYes, goexp)
    np_out = denorm2(npData_year, np_min, np_max, logaYes, goexp)
    scoring(ens_count, year, df_sum, np_out, domains, domIDs, "fileObs.csv");

    # log normalised values
    #goexp = +1
    np_out=np.copy(npData_year)
    #denorm(np_out, np_min, np_max, logaYes, goexp)
    scoring(ens_count, year, df_sum, np_out, domains, domIDs, "fileObsX.csv");

########################
#sys.exit();
if VERBOSE == 1:
    print(domains)
    print("domIDs")
    print(domIDs)
    print("scales")
    print(scales)

    print("titles")
    print(titles)
    print("entities")
    print(entities)

## back to dataFrame
df_init = pd.DataFrame(np_init, columns = titles)
df_init['Code']=codes;
df_init['Entity']=entities;
print(" df_head_latest: ");
print(df_init.head())

#df_init.to_csv("tmp1.csv", float_format="%.5f");


