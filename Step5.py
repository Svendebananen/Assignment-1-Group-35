# step 5: Balancing Market, no storage, no transmission, 1 hour
# imports
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt 

# import os library to consider the same folder of this file for the csv reading
import os
from pathlib import Path
os.chdir(Path(__file__).parent) 

# create output folder for plots
plots_dir = Path(__file__).parent / 'step 6 plots'
plots_dir.mkdir(exist_ok=True) 

# mute the gurobi license print
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start() 

class Expando(object):
    pass

#define classes for input data and optimization problem
class LP_InputData:

    def __init__(
        self, 
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],               # Coefficients in objective function
        constraints_coeff: dict[str, dict[str,float]],    # Linear coefficients of constraints
        constraints_rhs: dict[str, float],                # Right hand side coefficients of constraints
        constraints_sense: dict[str, int],              # Direction of constraints
        objective_sense: int,                           # Direction of optimization
        model_name: str                                 # Name of model
    ):
        self.VARIABLES = VARIABLES
        self.CONSTRAINTS = CONSTRAINTS
        self.objective_coeff = objective_coeff
        self.constraints_coeff = constraints_coeff
        self.constraints_rhs = constraints_rhs
        self.constraints_sense = constraints_sense
        self.objective_sense = objective_sense
        self.model_name = model_name

class LP_OptimizationProblem():
    def __init__(self, input_data: LP_InputData): # initialize class
        self.data = input_data # define data attributes
        self.results = Expando() # define results attributes
        self._build_model() # build gurobi model
    
    def _build_variables(self):
        self.variables = {v: self.model.addVar(lb=0, name=f'{v}') for v in self.data.VARIABLES}
    
    def _build_constraints(self):
        self.constraints = {c:
                self.model.addLConstr(
                        gp.quicksum(self.data.constraints_coeff[c].get(v, 0) * self.variables[v] for v in self.data.VARIABLES),
                        self.data.constraints_sense[c],
                        self.data.constraints_rhs[c],
                        name = f'{c}'
                ) for c in self.data.CONSTRAINTS
        }

    def _build_objective_function(self):
        objective = gp.quicksum(self.data.objective_coeff[v] * self.variables[v] for v in self.data.VARIABLES)
        self.model.setObjective(objective, self.data.objective_sense)

    def _build_model(self): 
        self.model = gp.Model(name=self.data.model_name, env=env)
        self.model.setParam('OutputFlag', 0)  # suppress Gurobi output
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()
    
    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.variables = {v.VarName:v.x for v in self.model.getVars()}
        self.results.optimal_duals = {f'{c.ConstrName}':c.Pi for c in self.model.getConstrs()}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {self.model.ModelName} was not successful")
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Social Welfare:", self.results.objective_value)
        for key, value in self.results.variables.items():
            label = key
            if key.startswith("production of generator "):
                suffix = key.split(" ")[-1]
                if suffix.isdigit():
                    label = f"production of generator {int(suffix) + 1}"
            elif key.startswith("demand of load "):
                suffix = key.split(" ")[-1]
                if suffix.isdigit():
                    label = f"demand of load {int(suffix) + 1}"
            print(f'Optimal value of {label}:', value)
        for key, value in self.results.optimal_duals.items():
            label = key
            if key.startswith("capacity constraint "):
                suffix = key.split(" ")[-1]
                if suffix.isdigit():
                    label = f"capacity constraint {int(suffix) + 1}"
            print(f'Dual variable of {label}:', value)




# Import data from case study
date = '2019-08-31' # Choose data for wind turbine generation

conventional_generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id','bus','capacity','cost']) # conventional generators data

# Creating the wind capacity matrix for 6 wind generators and 24 hours
wind_capacity= np.zeros((6, 24)) # placeholder for wind generator data, to be filled with actual data from CSV files for hourly optimization
file_list = glob.glob(r'Ninja\*.csv')
for i,csv in enumerate(file_list):
  data = pd.read_csv(csv, header = None,names = ['time','local_time','capacity_factor'], skiprows = 4)
  index = data.loc[data['time'] == date + ' 00:00'].index[0] # Find the index of the row corresponding to the specified date starting at 00:00
  wind_capacity[i,:] = data['capacity_factor'][index:index+24].values*200 

wind_generator = pd.DataFrame({ # wind generators data, with capacity to be updated for each hour based on CSV files
        'id': [f'wind_{i}' for i in range(wind_capacity.shape[0])],
        'bus': pd.read_csv('wind_farms.csv',usecols=['node'])['node'].values,
        'capacity': 0.0,# placeholder, will be updated for each hour
        'cost': [0.0 for i in range(wind_capacity.shape[0])]
    }) 

# Creating a single DataFrame with all generators (conventional + wind)
total_generators =  pd.concat([conventional_generators, wind_generator], ignore_index = True)

# Load data upload
loads = pd.read_csv('LoadData.csv', header = None, usecols = [1], names = ['demand'])  # load data (hourly system demand)
load_distribution = pd.read_csv('load_distribution_1.csv')  # nodal load shares
load_nodes = load_distribution['node'].tolist()  # list of all load nodes
load_percentages = dict(zip(load_distribution['node'], load_distribution['pct_of_system_load'] / 100))  # fraction of total demand per node

# List of elastic loads: nodes 1, 7, 9, 13, 14, 15
elastic_nodes = [1, 7, 9, 13, 14, 15]

# Bid prices for elastic loads (€/MWh) — differentiated, consistent with generation costs (€5.47–26.11/MWh)
elastic_bid_prices = {
    1:  12.0,   # price-sensitive industrial load
    7:  22.0,   # commercial load
    9:  10.0,   # very price-sensitive, curtails early
    13: 20.0,   # commercial/industrial mix
    14: 16.0,   # mid-range flexibility
    15: 25.0,   # less flexible, close to peak generator cost
}

# Hour selected for merit order curve analysis (0-based index)
hour = 9
t = 9   # hour 5


print('-------------------')
print('Day-Ahead Market Clearing for Hour', (t + 1))

print('-------------------')

# Define ranges and indexes
N_GENERATORS = len(total_generators) # number of generators (12 conventional + 6 wind)
N_LOADS = len(load_distribution) # number of loads (17 total: 11 inelastic + 6 elastic)
#time_step = 24 # time step in hours 
GENERATORS = range(len(total_generators)) 
LOADS = range(N_LOADS) 


#Clear Day Ahead market for given hour 
wind_generator['capacity'] = wind_capacity[:, t] # Update wind generator capacities for the current hour based on CSV data
total_generators =  pd.concat([conventional_generators, wind_generator], ignore_index=True) # Update total generators DataFrame with the new wind generator capacities for the current hour

total_demand = loads['demand'][t] # update total demand for the current hour
#print(total_demand)

# Create demand data for each of the 17 loads
bid_quantities_min = []
bid_quantities_max = []
bid_prices = []

for node in load_nodes:
    demand_at_node = total_demand * load_percentages[node]
    
    if node in elastic_nodes:
        bid_quantities_min.append(demand_at_node * 0.20)
        bid_quantities_max.append(demand_at_node * 1.10)
        bid_prices.append(elastic_bid_prices[node])
    else:
        bid_quantities_min.append(demand_at_node * 1.00)
        bid_quantities_max.append(demand_at_node * 1.00)
        bid_prices.append(300.0) # high bid price to ensure the inelastic load are always accepted

demand_data = pd.DataFrame({
    'node': load_nodes,
    'bid_quantity_min': bid_quantities_min,
    'bid_quantity_max': bid_quantities_max,
    'bid_price': bid_prices
})


input_data = {
    'model0': LP_InputData(
        VARIABLES = [f'production of generator {g}' for g in GENERATORS] + \
                    [f'demand of load {j}' for j in LOADS],

        CONSTRAINTS = ['balance constraint'] + \
                        [f'capacity constraint {g}' for g in GENERATORS] + \
                        [f'demand min limit {j}' for j in LOADS] + \
                        [f'demand max limit {j}' for j in LOADS],
        
        objective_coeff = {
            # Consumer utility (positive)
            **{f'demand of load {j}': demand_data['bid_price'][j] for j in LOADS},
            # Generation cost (negative)
            **{f'production of generator {g}': -total_generators['cost'][g] for g in GENERATORS}
        },
        
        constraints_coeff = {
            # Balance constraint: total generation must equal total demand
            'balance constraint': {
                **{f'demand of load {j}': 1 for j in LOADS},
                **{f'production of generator {g}': -1 for g in GENERATORS}
            },
            # Generator capacity
            **{f'capacity constraint {g}': {f'production of generator {k}': int(k == g) for k in GENERATORS} for g in GENERATORS},
            # Demand minimum limits
            **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
            # Demand maximum limits
            **{f'demand max limit {j}': {f'demand of load {j}': 1} for j in LOADS}
        },
        
        constraints_rhs = {
            'balance constraint': 0,
            **{f'capacity constraint {g}': total_generators['capacity'][g] for g in GENERATORS},
            **{f'demand min limit {j}': demand_data['bid_quantity_min'][j] for j in LOADS},
            **{f'demand max limit {j}': demand_data['bid_quantity_max'][j] for j in LOADS}
        },
        
        constraints_sense = { 
            'balance constraint': GRB.EQUAL,
            **{f'capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS},
            **{f'demand min limit {j}': GRB.GREATER_EQUAL for j in LOADS},
            **{f'demand max limit {j}': GRB.LESS_EQUAL for j in LOADS}
        },
        
        objective_sense = GRB.MAXIMIZE, # maximize social welfare
        model_name = "Market Clearing - 17 Loads (11 Inelastic + 6 Elastic)"
    )
}

model = LP_OptimizationProblem(input_data['model0'])
model.run()
mcp = model.results.optimal_duals['balance constraint']

print('Day Ahead Market Clearing Price', mcp)
print('-------------------')

total_demand_served = sum(model.results.variables[f'demand of load {j}'] for j in LOADS)
total_generators['scheduled_production'] = [ model.results.variables[f'production of generator {g}'] for g in GENERATORS]



########################################################################################################################################################
# BALANCING MARKET
########################################################################################################################################################
print('Balancing Market')
print('-------------------')

########################################################################################################################################################
# INPUT 
########################################################################################################################################################

generator_actual_production = { # actual production of each generator id, to be used for balancing market, based on the optimization results for the selected hour (hour 5)
    10:  0,   # failure 
    'wind_0':  1.1,   # higher production
    'wind_1':  1.1,   # 
    'wind_2': 0.85,   # lower
    'wind_3': 0.85,   # 
    'wind_4': 1.1,
    'wind_5': 0.85,
      
}

curtailment_cost = 500 # cost of curtailing load in the balancing market (€/MWh), set high to prioritize generation adjustments over load curtailment
potential_balancing_generators = [1,4,5,6,8,9] # list of generator ids that can provide balancing services (excluding the failed generator 10 and the wind generators with uncertain production)


########################################################################################################################################################
# DATA  
########################################################################################################################################################

#compute actual production for each generator based on the optimization results and the actual production factors, considering only the used generators and the capacity of each generator
total_generators['actual_production'] = (
    total_generators['id'].map(generator_actual_production)
        .fillna(1)
        *total_generators['scheduled_production']
)

total_load = total_demand_served

total_generation = total_generators['actual_production'].sum() # total generation based on actual production of each generator
system_imbalance = total_generation - total_load # difference between generation and load, to be balanced in the balancing market

balancing_data = total_generators[
    (total_generators['id'].isin(potential_balancing_generators))
].copy().reset_index()

balancing_data['down_capacity'] = balancing_data['actual_production']
balancing_data['up_capacity'] = np.where(
    (balancing_data['capacity'] - balancing_data['actual_production']) > 0,
    balancing_data['capacity'] - balancing_data['actual_production'],
    0
)

balancing_data['up_price'] = mcp * (1 + 0.10 * balancing_data['cost'])
balancing_data['down_price'] = mcp * (1 - 0.15 * balancing_data['cost'])

########################################################################################################################################################
# BALANCING MARKET IMBALANCE
########################################################################################################################################################


print('-------------------')
print('System imbalance',system_imbalance)

print('Total upwards capacity',balancing_data['up_capacity'].sum())
print('Total downwards capacity', balancing_data['down_capacity'].sum())


########################################################################################################################################################
# SOLVE BALANCING MARKET
########################################################################################################################################################

# Define ranges and indexes
N_GENERATORS = len(balancing_data) # number of generators (12 conventional + 6 wind)
 
GENERATORS = range(len(balancing_data)) 

input_data = {
    'model1': LP_InputData(
        VARIABLES = [f'up regulation {g}' for g in GENERATORS] + \
                    [f'down regulation {g}' for g in GENERATORS]+ \
                    ['curtailment'],

        CONSTRAINTS = ['balancing constraint'] + \
                        [f'up capacity constraint {g}' for g in GENERATORS] + \
                        [f'down capacity constraint {g}' for g in GENERATORS]+\
                        ['curtailment limit'],

        
        objective_coeff = {
            **{f'up regulation {g}': balancing_data['up_price'][g] for g in GENERATORS},
            **{f'down regulation {g}': -balancing_data['down_price'][g] for g in GENERATORS},
            'curtailment': curtailment_cost
        },

        constraints_coeff = {
            

            # Balancing constraint
            'balancing constraint': {
                **{f'up regulation {g}': 1 for g in GENERATORS},
                **{f'down regulation {g}': -1 for g in GENERATORS},
                'curtailment': 1
            },

            # Up regulation capacity
            **{f'up capacity constraint {g}': {
                f'up regulation {k}': int(k == g) for k in GENERATORS
            } for g in GENERATORS},

            # Down regulation capacity
            **{f'down capacity constraint {g}': {
                f'down regulation {k}': int(k == g) for k in GENERATORS
            } for g in GENERATORS},


            'curtailment limit': {'curtailment': 1}
        },

        constraints_rhs = {

            # System imbalance
            'balancing constraint': system_imbalance,

            # Upward regulation limits
            **{f'up capacity constraint {g}': balancing_data['up_capacity'][g] for g in GENERATORS},

            # Downward regulation limits
            **{f'down capacity constraint {g}': balancing_data['down_capacity'][g] for g in GENERATORS},

            'curtailment limit': total_load
        },



        constraints_sense = {

            'balancing constraint': GRB.EQUAL,

            **{f'up capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS},

            **{f'down capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS},

            'curtailment limit': GRB.LESS_EQUAL
        },

        objective_sense = GRB.MINIMIZE,

        model_name = "Balancing Market Clearing" )
    
}

model = LP_OptimizationProblem(input_data['model1'])
model.run()

########################################################################################################################################################
# RESULTS
########################################################################################################################################################

balancing_price = model.results.optimal_duals['balancing constraint']

balancing_results_down = [model.results.variables[f'down regulation {g}'] for g in GENERATORS]
balancing_results_up = [ model.results.variables[f'up regulation {g}'] for g in GENERATORS]
#balancing_results  = balancing_results_down-balancing_results_up
curtailment = [model.results.variables['curtailment']]
balancing_results = pd.DataFrame({'id': potential_balancing_generators,'balancing_down':balancing_results_down,'balancing_up':balancing_results_up})
print(len(curtailment))
total_generators = total_generators.merge(
    balancing_results[['id', 'balancing_up', 'balancing_down']],
    on='id',
    how='left'
).fillna(0).copy()



total_generators['production_imbalance'] =(total_generators['scheduled_production']-total_generators['actual_production'])
print(total_generators['production_imbalance'])
# One-price scheme profit
total_generators['profit_one_price'] = (total_generators['scheduled_production']*mcp+
                                        total_generators['production_imbalance']*balancing_price+
                                        
    total_generators['balancing_up'] * balancing_price
    - total_generators['balancing_down'] * balancing_price

)
#Two price scheme profit
total_generators['profit_two_price']=np.nan

if system_imbalance < 0:
    for g in range(len(total_generators)): 
        if total_generators['production_imbalance'][g]>0:
            two_price_scheme = total_generators['production_imbalance'][g]*mcp
        else:
            two_price_scheme =total_generators['production_imbalance'][g]*balancing_price

        total_generators['profit_two_price'][g] = (total_generators['scheduled_production'][g]*mcp+two_price_scheme+
        total_generators['balancing_up'][g] * balancing_price- total_generators['balancing_down'][g] * balancing_price)
else:
     for g in range(len(total_generators)): 
        if total_generators['production_imbalance'][g]>0:
            two_price_scheme = total_generators['production_imbalance'][g]*balancing_price
        else:
            two_price_scheme =total_generators['production_imbalance'][g]*mcp
        total_generators['profit_two_price'][g] = (total_generators['scheduled_production'][g]*mcp+two_price_scheme+
        total_generators['balancing_up'][g] * balancing_price- total_generators['balancing_down'][g] * balancing_price)

# Total system profit
#total_profit_one_price = total_generators['profit_one_price'].sum()-curtailment*curtailment_cost
#total_profit_two_price = total_generators['profit_two_price'].sum()
# print("Total Profit (One-Price Scheme):", total_profit_one_price)
# print("Total Profit (Two-Price Scheme):", total_profit_two_price)


curtailment_cost