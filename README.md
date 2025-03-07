<h2> MOE (Model Of Everything) </h2>

Purpose: Simulates state indicators for a bunch of countries from 1991 to 2017.
<br>
Disclimer: THis code is intended to facilitate discussion and stimulate hypotheses concerning scenarios.
By no means it can be used to support management practices or advise policy decisions.
<br><br>
HOWTO:
<ol>
<li> Open wbank_cov.py, search for USER INPUT, uncomment set of countries to simulate over, run the script to produce error-covariance:

python3 wbank_cov.py 
</li><li>
Open wbank_das.py, search for USER INPUT, uncomment one of the simulation scenarios, run the script:

python3 wbank_das.py
</li><li>
Io visualise simulated data:

python3 wbank_plot_ts_var.py

(red is observations)
</li>
</ol>
 
Author: Nugzar Margvelashvili

Hobart, December 2024.

 
Reference: to be added when published

####################
<br><br>
Further Details:

<br>
a) Data

The Number of Battle Related Deaths (State Conflicts) have been obtained from Uppsala Data Conflict Program
(https://ucdp.uu.se/ )

The rest of the data (coming from various sources) have been formatted to a uniform style and are available online from “Our World in Data” (https://ourworldindata.org/ ). 

<br>
b) Model

THe model is based on an ensemble optimal imterpolation. Each country is defined as a collection of state indicators such as area, population, GDP, CO2 emissions, inequality, life expectancy, among others. For each simulated year and each country the model assumes state indicators of that coutry next year are 
exactly the same as the state indicators this year except random perturbations. 
These perturbations have a Gaussian probability distribution and parameters of that distribution (mean and covariance matrix) are derived from historical observations. 
The covariance matrix holds an information about correlations between these perturbations. Given these (and a couple of other) assumptions, 
and assuming the value of an "observed" state indicator "A" is known, one can make an educated guess of the value of another "unobserved" state indicator "B".

<br>
c) Plots

Every plot shows an index of either a specific state-indicator or an index aggregated over a disciplinaty domain and integrated over the globe.
Indices for contaminants and conflict-deaths vary from -1 to 0. All other indices vary from 0 to 1. 
The higher the value of the index the better the quality of the indicator.


