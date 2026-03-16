'''This file loads the data for the case study, including generator data, wind capacity, load data, and elastic load bid prices. 
It prepares the data for use in the optimization problems in subsequent steps.'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# importing os library to consider the same folder of this file for the csv reading
from pathlib import Path
####################################################################################################################################################################
#  USER INPUTS
#####################################################################################################################################################################
DATE = '2019-08-31' # Choose data for wind turbine generation
HOUR = 8            # Choose hour of the day for optimization (0-23)
####################################################################################################################################################################

conventional_generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id','bus','capacity','cost']) # conventional generators data

# Creating the wind capacity matrix for 6 wind generators and 24 hours
wind_capacity= np.zeros((6, 24)) # placeholder for wind generator data, to be filled with actual data from CSV files for hourly optimization
file_list = sorted(Path(__file__).parent.glob('Ninja/*.csv'))
for i,csv in enumerate(file_list):
  data = pd.read_csv(csv, header = None,names = ['time','local_time','capacity_factor'], skiprows = 4)
  index = data.loc[data['time'] == DATE + ' 00:00'].index[0] # Find the index of the row corresponding to the specified date starting at 00:00
  wind_capacity[i,:] = data['capacity_factor'][index:index+24].values*200 

wind_generator = pd.DataFrame({ # wind generators data, with capacity to be updated for each hour based on CSV files
        'id': [f'wind_{i}' for i in range(wind_capacity.shape[0])],
        'bus': pd.read_csv('wind_farms.csv',usecols=['node'])['node'].values,
        'capacity': 0.0,# placeholder, will be updated for each hour
        'cost': [0.0 for i in range(wind_capacity.shape[0])]
    }) 
wind_generator['capacity'] = wind_capacity[:, HOUR] # update wind generator capacities for the selected hour
# Creating a single DataFrame with all generators (conventional + wind)
total_generators =  pd.concat([conventional_generators, wind_generator], ignore_index = True)

# Load data upload
loads = pd.read_csv('LoadData.csv', header = None, usecols = [1], names = ['demand'])  # load data (hourly system demand)
load_distribution = pd.read_csv('load_distribution_1.csv')  # nodal load shares
load_nodes = load_distribution['node'].tolist()  # list of all load nodes
load_percentages = dict(zip(load_distribution['node'], load_distribution['pct_of_system_load'] / 100))  # fraction of total demand per node

#define the path and clear eventual spaces in the csv
df = pd.read_csv('elastic_data.csv')
df.columns = df.columns.str.strip()

# List of elastic loads: nodes 1, 7, 9, 13, 14, 15
elastic_nodes = df['node'].tolist()

# Bid prices for elastic loads (€/MWh) — differentiated, consistent with generation costs (€5.47–26.11/MWh)
elastic_bid_prices = elastic_bid_prices = df.set_index('node')['bid'].to_dict()
VOLL = 500  

generator_actual_production = { # actual production of each generator id, to be used for balancing market, based on the optimization results for the selected hour (hour 5)
    10:  0,   # failure 
    'wind_0':  1.1,   # higher production
    'wind_1':  1.1,   # 
    'wind_2': 0.85,   # lower
    'wind_3': 0.85,   # 
    'wind_4': 1.1,
    'wind_5': 0.85,
      
}

CURTAILMENT_COST = 500 # cost of curtailing load in the balancing market (€/MWh), set high to prioritize generation adjustments over load curtailment
potential_balancing_generators = [1,4,5,6,8,9] # list of generator ids that can provide balancing services (excluding the failed generator 10 and the wind generators with uncertain production)

