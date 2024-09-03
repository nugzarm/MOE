MOE (Model Of Everything)

Purpose: Simulates state indicators for a bunch of countries using ensemble optimal interpolation.

Disclimer: THis code is intended to facilitate discussion and stimulate hypotheses concerning future scenarios.
By no means it can be used to support management practices or advise policy decisions.

HOWTO run:

#Step1:  open wbank_cov.py, search for USER INPUT, uncomment set of countries to simulate over, run the script to produce error-covariance:

python3 wbank_cov.py 

#Step2:  open wbank_das.py, search for USER INPUT, uncomment one of the simulation scenarios, run the script:

python3 wbank_das.py

#Step 3:  to visualise simulated data:

python3 wbank_plot_ts_var.py

(red is observations)

#To run on NCI:

module use /g/data/up99/modulefiles

module load NCI-geophys/23.03

 

Author: Nugzar Margvelashvili

Hobart, August 2024.

 
Reference: to be added when published

####################

Each country is defined as a collection of state indicators such as area, population, GDP, CO2 emissions, inequality, life expectancy, among others. These indicators collectively characterize a country and often vary over time. To analyse these variations, we assume a Gaussian multivariate distribution of these indicators with known mean and error-covariance derived from data. This enables us to build a numerical model evolving the state indicators through time. Despite relatively simple formulation, this model shows a remarkable agreement with observations which has prompted siulation of several “what if …” scenarios.

To analyse the model output, a multiplicative factor of either (+1) or (-1) is assigned to every state indicator thus transforming it into the “quality” index (all contaminants and deaths indicators having a negative “quality” index).  A total state “quality” index is calculated as an arithmetic mean of the “quality” indexes of individual indicators. A similar “quality” index is calculated for subgroups of indicators representing environmental contaminants, agriculture, economic development, technology and innovation, social progress, and social disruptions given by violent deaths (and encompassing both homicide and battle-related deaths in state conflicts). The “quality” index varies from -1 to 0 for the negative indicators, and from 0 to 1 for the positive indicators, the higher the value the better the “quality” of the indicator.


Data

The Number of Battle Related Deaths (State Conflicts) have been obtained from Uppsala Data Conflict Program
(https://ucdp.uu.se/ )

The rest of the data (coming from various sources) have been formatted to a uniform style and offered online by “Our World in Data” (https://ourworldindata.org/ ). 

#########################################
