#importing libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
#from mpl_toolkits.basemap import Basemap
import os
os.environ['USE_PYGEOS'] = '0'

import geopandas
import folium
import sys;
import math;

fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)


fname = ""
file_x='states.csv'
fx = open(file_x);

#https://www.learnpythonwithrune.org/plot-world-data-to-map-using-python-in-3-easy-steps/

#
file_states='states.csv'
states_pd = pd.read_csv(file_states, sep=',', header=0)
#print(states_pd.head())
states = states_pd.values.flatten();
#for state in states:
#    print(state)

file_codes='codes.csv'
codes_pd = pd.read_csv(file_codes, sep=',', header=0)
#print(states_pd.head())
codes = codes_pd.values.flatten();




file_brd='nm_GEDEvent_v22_1.csv'
brd = pd.read_csv(file_brd, sep=',', header=0)
brd = brd.dropna(subset=['Code'])
#pop['Population']=(pop['Population'] - pop['Population'].min())/(pop['Population'].max() - pop['Population'].min())
#print("Population")
print("brd_haed")
print(brd.head())
print("brd_tail")
print(brd.tail())

for code in codes:
    print(code)
    brd_sum = brd.groupby(['Code', 'Year'])['brd_best'].sum()

print(brd_sum.head())

# dump to disk
#brd_sum.to_csv("brd_sum.ts");

scale=1
colName="GDPPC2015"
fileName="nm_gdp-per-capita-in-us-dollar-world-bank.csv"


colName="LandArea_sq_km"
fileName="land-area-km.csv"

pop = pd.read_csv(fileName, sep=',', header=0)
pop = pop.dropna(subset=['Code'])
#pop = pop.drop(columns=['Entity'])
pop[colName]=scale*(pop[colName] - pop[colName].min())/(pop[colName].max() - pop[colName].min())
print(colName, pop.shape)
print(colName, pop.head())

result=pop.copy(deep=True);
result = pd.merge(result, brd_sum, how="left", on=["Code", "Year"])
#result = result.dropna(subset=['ForestArea'])


print("result", result.head())
result.to_csv("battle_death.ts", na_rep='3');


sys.exit();

#########################################

# dump to disk
#result.to_csv("result.ts");

# plot
# Read the geopandas dataset
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
print(world.head())

world.head()


# Merge the two DataFrames together
brd = world.merge(brd, how="left", left_on=['iso_a3'], right_on=['Code'])
# Merge the two DataFrames together
#happy = world.merge(happy, how="left", left_on=['iso_a3'], right_on=['Code'])

# filter a year
nameVar='brd_best'
showVar = brd.copy()


is_2018 = showVar['Year']==2018
showVar_2018=showVar[is_2018]


# Clean data: remove rows with no data
#gdp = gdp.dropna(subset=['GDP_2015_USD'])

# Create a map
my_map = folium.Map()
# Add the data
folium.Choropleth(
    geo_data=showVar,
    name='choropleth',
    #data=happy.loc[(happy['Year']=='2018')],
    data=showVar_2018,
    #columns=['Entity', 'GDP_2015_USD'],
    columns=['Code', nameVar],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    nan_fill_color="White",
    fill_opacity=0.71,
    line_opacity=0.2,
    legend_name=nameVar
).add_to(my_map)
my_map.save(nameVar+'.html')



sys.exit();

#df = pd.read_csv(file_coast, sep='\s+', header=None)
#array = df.apply(pd.to_numeric, args=('coerce',)).values

imax = +350;


def animate(ii):

    i = ii+200;

    print("i====",i);
    if(i > imax):
        sys.exit();


    #data_x = open('../out_grain/grain_x0','r').read()
    #lines_x = data_x.split('\n').pop(0)
    #data_y = open('../out_grain/grain_y0','r').read()
    #lines_y = data_y.split('\n').pop(0)


    #fx = open(file_x);
    #fy = open(file_y);

    lines_x='';
    lines_y='';
    lines_z='';

    fx.seek(1);
    fy.seek(1);
    fz.seek(1);
    
    # lines to print
    specified_lines = int(i)+1;
    # loop over lines in a file
    for pos, lines_x in enumerate(fx):
        # check if the line number is specified in the lines to read array
        if int(pos) == specified_lines:
            # print the required line number
            #print("x: ", lines_x)
            break;

    # loop over lines in a file
    for pos, lines_y in enumerate(fy):
        # check if the line number is specified in the lines to read array
        if int(pos) == specified_lines:
            # print the required line number
            #print("y: ", lines_y)
            break;

    # loop over lines in a file
    for pos, lines_z in enumerate(fz):
        # check if the line number is specified in the lines to read array
        if int(pos) == specified_lines:
            # print the required line number
            #print("mass: ", lines_m)
            break;


    #print("bubu", lines_y)

    xs = []
    ys = []
    zs = []
    xsf = []
    ysf = []
    zsf = []
  
    # split the string
    xs = lines_x.split();
    ys = lines_y.split();
    zs = lines_z.split();
    # get rid of the date
    del xs[0]
    del ys[0]
    del zs[0]
    # convert to float
    #xsf = [1*float(item) for item in xs]
    #ysf = [1*float(item) for item in ys]

    ysf = [ float(item)  for item in lat]
    xsf = [ float(item)  for item in lon]
    #zsf = [ float(item)  for item in zs]
    #zsf = [min(float(item), 1) for item in zs]
    zsf = [math.log(float(item)+1e-6, 10) for item in zs]

    print(len(xsf), len(ysf), len(zsf))

    #for k in range(len(xsf)):
    #    xsf[k] = xsf[k]/(math.cos(ysf[k]))

    #print(xsf)
    #print(ysf)
    
    ax1.clear()
    #ax1.set(xlim=(-3.15, 3.15), ylim=(-1.6, 1.6))
    ax1.set(xlim=(1.9, 2.8),ylim=(-0.9, 0)) # ip

    #ax1.scatter(xsf, ysf, marker='o', s=5)
    #ax1.scatter(xsf, ysf, c=msf, s=5, cmap='jet')
    #ax1.scatter(xsf, ysf, c=zsf, s=5, cmap='jet')
    sc = ax1.scatter(xsf, ysf, c=zsf, s=5, cmap='jet', vmin=-5, vmax=-1);
    #sc = ax1.scatter(xsf, ysf, c=zsf, s=5, cmap='terrain', vmin=-5, vmax=-0);
    #sc = ax1.scatter(xsf, ysf, c=zsf, s=5, cmap='jet')

    #ax1.plot(lon, lat, color='k', linewidth=0.2)

    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('time = '+str(i) )  
    cb = plt.colorbar(sc)
    plt.pause(0.01)
    cb.remove()
	
    #fx.close()
    #fy.close()

#animate(3)

plt.show()



