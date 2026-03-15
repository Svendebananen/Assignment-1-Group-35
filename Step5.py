'''STEP 5 - Balancing Market Clearing (1 hour, no storage, no transmission)
Using Day Ahead Market results for hour 9 from step 1,  compute the actual production of each generator based on the actual production factors,
and solve the balancing market clearing problem to find the balancing price and the optimal up/down regulation for each generator. 
Then, compute the profit for each generator under a one-price scheme and a two-price scheme, and compare the results.'''

########################################################################################################################################################
# IMPORTS LIBRARIES,MODULES AND DATA
########################################################################################################################################################
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from module_LP import LP_InputData, LP_OptimizationProblem, Expando
os.chdir(Path(__file__).parent)

plots_dir = Path(__file__).parent / 'step 5 plots'
plots_dir.mkdir(exist_ok=True)#
from Assignment_step1 import (production_t, total_demand_served_t, mcp_t)
from load_casestudy import (total_generators, generator_actual_production,potential_balancing_generators,CURTAILMENT_COST)

mcp = mcp_t

total_load = total_demand_served_t
day_ahead_results = total_generators.copy().drop(columns=['bus'])
day_ahead_results['scheduled_production'] = production_t

########################################################################################################################################################
# DAY AHEAD RESULTS AND BALANCING MARKET DATA PREPARATION
########################################################################################################################################################

#compute actual production for each generator based on the optimization results and the actual production factors, 
day_ahead_results['actual_production'] = (
    day_ahead_results['id'].map(generator_actual_production)
        .fillna(1)
        *day_ahead_results['scheduled_production']
)

total_generation = day_ahead_results['actual_production'].sum() # total generation based on actual production
system_imbalance = total_load -  total_generation # difference between generation and load, to be balanced in the balancing market

print('Day Ahead Results with Actual Production')
print('-------------------')
print(day_ahead_results)

#build Dataframe for balancing market optimization, with up/down regulation capacities and prices for each generator based on the actual production
balancing_data = day_ahead_results[
    (day_ahead_results['id'].isin(potential_balancing_generators))
].copy().reset_index()

balancing_data['down_capacity'] = balancing_data['actual_production']
balancing_data['up_capacity'] = np.where(
    (balancing_data['capacity'] - balancing_data['actual_production']) > 0,
    balancing_data['capacity'] - balancing_data['actual_production'],
    0)

balancing_data['up_price'] = np.maximum((mcp + 0.10 * balancing_data['cost']),mcp)
balancing_data['down_price'] = mcp - 0.15 * balancing_data['cost']



curtailment_data = pd.DataFrame({"id": ["load"], "capacity": [total_load], "cost": [CURTAILMENT_COST], "scheduled_production":[total_load],"actual_production":[total_load],"up_capacity": [total_load], "down_capacity": [0], "up_price": [CURTAILMENT_COST], "down_price": [0]})
balancing_data = pd.concat([balancing_data, curtailment_data], ignore_index=True)
balancing_data = balancing_data[["id","up_capacity", "down_capacity", "up_price", "down_price"]]
print('Balancing Market Data')
print('-------------------')
print(balancing_data)

########################################################################################################################################################
# SYSTEM IMBALANCE AND BALANCING CAPACITY
########################################################################################################################################################

print('-------------------')
print('System imbalance',system_imbalance)
if system_imbalance > 0:
    print('System needs upward regulation to increase generation or reduce load')
    print('Upwards capacity', balancing_data['up_capacity'][:-1].sum())
elif system_imbalance < 0:
    print('System needs downward regulation to reduce generation or increase load')
    print('Total downwards capacity', balancing_data['down_capacity'].sum())




# ########################################################################################################################################################
# # SOLVE BALANCING MARKET
# ########################################################################################################################################################

# Define ranges and indexes
N_GENERATORS = len(balancing_data)  
GENERATORS = range(len(balancing_data)) 

input_data = {
    'model1': LP_InputData(
        VARIABLES = [f'up regulation {g}' for g in GENERATORS] + \
                    [f'down regulation {g}' for g in GENERATORS],

        CONSTRAINTS = ['balancing constraint'] + \
                        [f'up capacity constraint {g}' for g in GENERATORS] + \
                        [f'down capacity constraint {g}' for g in GENERATORS],

        
        objective_coeff = {
            **{f'up regulation {g}': balancing_data['up_price'][g] for g in GENERATORS},
            **{f'down regulation {g}': -balancing_data['down_price'][g] for g in GENERATORS}
        },

        constraints_coeff = {
            

            # Balancing constraint
            'balancing constraint': {
                **{f'up regulation {g}': 1 for g in GENERATORS},
                **{f'down regulation {g}': -1 for g in GENERATORS}
            },

            # Up regulation capacity
            **{f'up capacity constraint {g}': {
                f'up regulation {k}': int(k == g) for k in GENERATORS
            } for g in GENERATORS},

            # Down regulation capacity
            **{f'down capacity constraint {g}': {
                f'down regulation {k}': int(k == g) for k in GENERATORS
            } for g in GENERATORS}
        },

        constraints_rhs = {

            # System imbalance
            'balancing constraint': system_imbalance,

            # Upward regulation limits
            **{f'up capacity constraint {g}': balancing_data['up_capacity'][g] for g in GENERATORS},

            # Downward regulation limits
            **{f'down capacity constraint {g}': balancing_data['down_capacity'][g] for g in GENERATORS}
        },



        constraints_sense = {

            'balancing constraint': GRB.EQUAL,

            **{f'up capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS},

            **{f'down capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS}
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

print('Balancing price:', balancing_price)

balancing_results_down = [model.results.variables[f'down regulation {g}'] for g in GENERATORS]
balancing_results_up = [ model.results.variables[f'up regulation {g}'] for g in GENERATORS]

balancing_results = pd.DataFrame({'id': potential_balancing_generators,'balancing_down':balancing_results_down[:-1],'balancing_up':balancing_results_up[:-1]})
balancing_results['balancing_service'] = balancing_results['balancing_up'] - balancing_results['balancing_down']
curtailment = balancing_results_up[-1] # load curtailment in the balancing market, if any
balancing_market = day_ahead_results.merge(
    balancing_results,
    on='id',
    how='left'
).fillna(0).copy()
balancing_market['imbalance'] = (
    balancing_market['actual_production']
    - balancing_market['scheduled_production']
)
print('Balancing Market Results')
print('-------------------')
print(balancing_market)
print('--------------------')
print('Curtailment in the balancing market (MWh):', curtailment)

# total_generators['production_imbalance'] =(total_generators['scheduled_production']-total_generators['actual_production'])
# print(total_generators['production_imbalance'])
# One-price scheme profit
balancing_market['profit_one_price'] = (balancing_market['scheduled_production']*mcp
                                        +balancing_market['balancing_service']*balancing_price
                                        + balancing_market['imbalance'] * balancing_price
                                        - balancing_market['cost'] * balancing_market['actual_production'])                                       

# #Two price scheme profit

# Default settlement price = MCP
balancing_market['imbalance_price'] = mcp

print('IMBALANCE',balancing_market['imbalance'])

# 3. Identify generators helping the system
if system_imbalance > 0:  # upward regulation needed, so generators providing upward regulation are helping the system
    helping_mask = balancing_market['imbalance'] > 0
else:                     #  downward regulation, needed, so generators providing downward regulation are helping the system
    helping_mask = balancing_market['imbalance'] < 0

# 4. Generators helping the system get balancing price
balancing_market.loc[helping_mask, 'imbalance_price'] = balancing_price


# 6. Profit calculation
balancing_market['profit_two_price'] = (
    balancing_market['scheduled_production'] * mcp
    + balancing_market['balancing_service'] * balancing_price
    + balancing_market['imbalance'] * balancing_market['imbalance_price']
    - balancing_market['cost'] * balancing_market['actual_production']
)

# Total system profit
total_profit_one_price = balancing_market['profit_one_price'].sum()-curtailment*CURTAILMENT_COST
total_profit_two_price = balancing_market['profit_two_price'].sum()-curtailment*CURTAILMENT_COST
print("Profit (One-Price Scheme):", balancing_market['profit_one_price'])
print("Profit (Two-Price Scheme):", balancing_market['profit_two_price'])
