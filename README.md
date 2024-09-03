Model: MOE (Model of Everything)

Purpose: Simulates state indicators for a bunch of countries individually using ensemble optimal interpolation.

Disclimer: THis code is intended to facilitate discussion and stimulate hypotheses concerning future scenarios.
By no means this code can be used to support management practices or advise policy decisions.

HOWTO to run:

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

#############################################################
