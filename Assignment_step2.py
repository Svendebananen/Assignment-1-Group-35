#Step 2: Include Storage and optimize over 24h with varying wind generation and elastic demand

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path




# FUNCTIONS


def build_hourly_gen_capacity(conventional_generators, wind_capacity):
    """Return a dict of generator capacities for each hour t"""
    gen_capacity = {}
    for t in T:
        cap_t = list(conventional_generators['capacity'].values) + list(wind_capacity[:, t])
        gen_capacity[t] = cap_t
    return gen_capacity

def build_hourly_demand_data(load_nodes, load_percentages, elastic_nodes, elastic_bid_prices, loads):
    """Return hourly demand data dict for each hour t"""
    hourly_demand_data = {}
    for t in T:
        total_demand = loads['demand'][t]
        bid_q_min, bid_q_max, bid_price = [], [], []
        for node in load_nodes:
            d = total_demand * load_percentages[node]
            if node in elastic_nodes:
                bid_q_min.append(d * 0.20)
                bid_q_max.append(d * 1.10)
                bid_price.append(elastic_bid_prices[node])
            else:
                bid_q_min.append(d)
                bid_q_max.append(d)
                bid_price.append(300.0)
        hourly_demand_data[t] = {
            'bid_quantity_min': bid_q_min,
            'bid_quantity_max': bid_q_max,
            'bid_price': bid_price,
            'total_demand': total_demand
        }
    return hourly_demand_data

def build_variables(GENERATORS, LOADS, T):
    """Return list of variable names"""
    VARIABLES = (
        [f'production of generator {g}_{t}' for g in GENERATORS for t in T] +
        [f'demand of load {j}_{t}' for j in LOADS for t in T] +
        [f'charge storage {t}' for t in T] +
        [f'discharge storage {t}' for t in T] +
        [f'energy storage {t}' for t in T]
    )
    return VARIABLES

def build_objective_coeff(VARIABLES, GENERATORS, LOADS, T, gen_costs, hourly_demand_data):
    """Return objective coefficients dictionary (social welfare)"""
    objective_coeff = {}
    for t in T:
        for j in LOADS:
            objective_coeff[f'demand of load {j}_{t}'] = hourly_demand_data[t]['bid_price'][j]
        for g in GENERATORS:
            objective_coeff[f'production of generator {g}_{t}'] = -gen_costs[g]
    # Storage has zero bid/offers
    for t in T:
        objective_coeff[f'charge storage {t}'] = 0
        objective_coeff[f'discharge storage {t}'] = 0
        objective_coeff[f'energy storage {t}'] = 0
    return objective_coeff

def build_constraints(GENERATORS, LOADS, T, gen_capacity, hourly_demand_data, Pch, Pdis, Emax, E0, eta_ch, eta_dis):
    constraints_coeff = {}
    constraints_rhs = {}
    constraints_sense = {}
    CONSTRAINTS = []

    # --- Balance constraints ---
    for t in T:
        cname = f'balance constraint {t}'
        CONSTRAINTS.append(cname)
        constraints_coeff[cname] = {
            **{f'demand of load {j}_{t}': 1 for j in LOADS},
            **{f'production of generator {g}_{t}': -1 for g in GENERATORS},
            f'charge storage {t}': -1,
            f'discharge storage {t}': 1
        }
        constraints_rhs[cname] = 0
        constraints_sense[cname] = GRB.EQUAL

    # --- Generator capacity ---
    for t in T:
        for g in GENERATORS:
            cname = f'capacity constraint {g}_{t}'
            CONSTRAINTS.append(cname)
            constraints_coeff[cname] = {f'production of generator {k}_{t}': int(k==g) for k in GENERATORS}
            constraints_rhs[cname] = gen_capacity[t][g]
            constraints_sense[cname] = GRB.LESS_EQUAL

    # --- Demand limits ---
    for t in T:
        for j in LOADS:
            cname_min = f'demand min {j}_{t}'
            cname_max = f'demand max {j}_{t}'
            CONSTRAINTS += [cname_min, cname_max]

            constraints_coeff[cname_min] = {f'demand of load {j}_{t}': 1}
            constraints_rhs[cname_min] = hourly_demand_data[t]['bid_quantity_min'][j]
            constraints_sense[cname_min] = GRB.GREATER_EQUAL

            constraints_coeff[cname_max] = {f'demand of load {j}_{t}': 1}
            constraints_rhs[cname_max] = hourly_demand_data[t]['bid_quantity_max'][j]
            constraints_sense[cname_max] = GRB.LESS_EQUAL

    # --- Storage limits ---
    for t in T:
        # Charge / Discharge limits
        for cname, var, limit in [(f'charge limit {t}', f'charge storage {t}', Pch),
                                   (f'discharge limit {t}', f'discharge storage {t}', Pdis),
                                   (f'energy max {t}', f'energy storage {t}', Emax)]:
            CONSTRAINTS.append(cname)
            constraints_coeff[cname] = {var: 1}
            constraints_rhs[cname] = limit
            constraints_sense[cname] = GRB.LESS_EQUAL

    # --- Storage intertemporal dynamics ---
    # Initial hour
    cname0 = 'storage dynamics 0'
    CONSTRAINTS.append(cname0)
    constraints_coeff[cname0] = {'energy storage 0': 1, 'charge storage 0': -eta_ch, 'discharge storage 0': 1/eta_dis}
    constraints_rhs[cname0] = E0
    constraints_sense[cname0] = GRB.EQUAL

    # t = 1..23
    for t in range(1, 24):
        cname = f'storage dynamics {t}'
        CONSTRAINTS.append(cname)
        constraints_coeff[cname] = {
            f'energy storage {t}': 1,
            f'energy storage {t-1}': -1,
            f'charge storage {t}': -eta_ch,
            f'discharge storage {t}': 1/eta_dis
        }
        constraints_rhs[cname] = 0
        constraints_sense[cname] = GRB.EQUAL

    return CONSTRAINTS, constraints_coeff, constraints_rhs, constraints_sense
def extract_hourly_results(model, GENERATORS, LOADS, load_nodes, elastic_nodes, demand_data, total_generators, t):
    """Extract all relevant metrics for a single hour"""
    
    # Market clearing price
    mcp = model.results.optimal_duals[f'balance constraint {t}']
    
    # Generation and demand
    total_generation = sum(model.results.variables[f'production of generator {g}_{t}'] for g in GENERATORS)
    total_demand_served = sum(model.results.variables[f'demand of load {j}_{t}'] for j in LOADS)
    total_battery_charge = model.results.variables[f'charge storage {t}']
    total_battery_discharge = model.results.variables[f'discharge storage {t}'] 
    total_battery_energy = model.results.variables[f'energy storage {t}']   
    
    # Elastic vs inelastic served
    elastic_demand_base = sum(demand_data[t]['bid_quantity_max'][j] / 1.10 for j, node in enumerate(load_nodes) if node in elastic_nodes)
    inelastic_demand_base = sum(demand_data[t]['bid_quantity_max'][j] for j, node in enumerate(load_nodes) if node not in elastic_nodes)
    
    elastic_served = sum(model.results.variables[f'demand of load {j}_{t}'] for j, node in enumerate(load_nodes) if node in elastic_nodes)
    inelastic_served = sum(model.results.variables[f'demand of load {j}_{t}'] for j, node in enumerate(load_nodes) if node not in elastic_nodes)
    
    # Costs
    total_cost = sum(total_generators['cost'][g] * model.results.variables[f'production of generator {g}_{t}'] for g in GENERATORS)
    consumer_utility = sum(demand_data[t]['bid_price'][j] * model.results.variables[f'demand of load {j}_{t}'] for j in LOADS)
    producer_surplus = mcp * total_generation - total_cost
    social_welfare = model.results.objective_value
    
    # Elastic flexibility
    elastic_flexibility_pct = (elastic_served / elastic_demand_base - 1) * 100 if elastic_demand_base > 0 else 0
    elastic_curtailment_from_max = (1 - elastic_served / (elastic_demand_base * 1.10)) * 100 if elastic_demand_base > 0 else 0
    
    return {
        'hour': t + 1,
        'mcp': mcp,
        'generation': total_generation,
        'demand_total': total_demand_served,
        'demand_inelastic': inelastic_served,
        'demand_elastic': elastic_served,
        'elastic_base': elastic_demand_base,
        'elastic_flexibility_pct': elastic_flexibility_pct,
        'elastic_curtailment_from_max': elastic_curtailment_from_max,
        'social_welfare': social_welfare,
        'total_cost': total_cost,
        'consumer_utility': consumer_utility,
        'producer_surplus': producer_surplus,
        'battery_charge': total_battery_energy,
    }
def plot_24h_results(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # MCP
    axes[0, 0].plot(results_df['hour'], results_df['mcp'], marker='o', color='blue', linewidth=2)
    axes[0, 0].set_title('Market Clearing Price (€/MWh)')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True)

    # Social Welfare
    axes[0, 1].plot(results_df['hour'], results_df['social_welfare'], marker='s', color='green', linewidth=2)
    axes[0, 1].set_title('Social Welfare (€)')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].grid(True)

    # Demand components
    axes[1, 0].plot(results_df['hour'], results_df['demand_inelastic'], marker='^', color='orange', label='Inelastic')
    axes[1, 0].plot(results_df['hour'], results_df['demand_elastic'], marker='v', color='purple', label='Elastic')
    axes[1, 0].plot(results_df['hour'], results_df['demand_total'], marker='o', color='red', linestyle='--', label='Total')
    axes[1, 0].set_title('Demand Components (MW)')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Elastic flexibility
    flex_values = results_df['elastic_flexibility_pct'].values
    colors = ['green' if x>0 else 'red' for x in flex_values]
    axes[1, 1].bar(results_df['hour'], flex_values, color=colors, alpha=0.7)
    axes[1, 1].axhline(0, color='black', linewidth=1)
    axes[1, 1].set_title("Elastic Demand Flexibility (%)")
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.show()
def plot_merit_order(load_nodes, elastic_nodes, elastic_bid_prices, loads, hour, results_df, conventional_generators, wind_generator, wind_capacity):
    # Supply
    moc_total_demand = loads['demand'][hour]
    wind_generator['capacity'] = wind_capacity[:, hour]
    moc_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)
    moc_generators_sorted = moc_generators.sort_values('cost')

    supply_cumulative = [sum(moc_generators_sorted['capacity'][:i]) for i in range(len(moc_generators_sorted))]
    supply_cost = list(moc_generators_sorted['cost'])

    # Demand
    demand_bids = []
    for node in load_nodes:
        qty = moc_total_demand * load_percentages[node]
        if node in elastic_nodes:
            bid = elastic_bid_prices[node]
            demand_bids.append((qty * 1.10, bid))
        else:
            demand_bids.append((qty, 300.0))

    demand_bids.sort(key=lambda x: x[1], reverse=True)
    demand_cumulative = [0]
    for i, (qty, _) in enumerate(demand_bids):
        demand_cumulative.append(demand_cumulative[-1] + qty)

    # Plot
    moc_equilibrium = results_df.loc[results_df['hour'] == hour + 1, 'mcp'].values[0]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.step(supply_cumulative, supply_cost, where='post', color='steelblue', linewidth=2.5, label='Supply Curve')
    ax.fill_between(supply_cumulative, supply_cost, step='post', alpha=0.2, color='steelblue')

    # Demand curve
    demand_x = []
    demand_y = []
    for i, (qty, bid) in enumerate(demand_bids):
        demand_x.append(demand_cumulative[i])
        demand_y.append(bid)
        demand_x.append(demand_cumulative[i+1])
        demand_y.append(bid)
    demand_x.append(demand_cumulative[-1])
    demand_y.append(0)
    ax.plot(demand_x, demand_y, color='red', linewidth=2.5, label='Demand Curve')

    # MCP line
    ax.axhline(moc_equilibrium, color='green', linestyle='--', linewidth=2, label=f'MCP: €{moc_equilibrium:.2f}/MWh')

    ax.set_xlabel('Cumulative Capacity / Demand (MW)')
    ax.set_ylabel('Price (€/MWh)')
    ax.set_title(f'Merit Order & Demand Curve - Hour {hour + 1}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
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
        self.model = gp.Model(name=self.data.model_name)
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
class Expando(object):
    pass





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

# creating a single DataFrame with all generators (conventional + wind)
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

#define parameters

T = range(24)  # 24 hours
GENERATORS = range(len(total_generators))  # all generators (conv + wind)
LOADS = range(len(load_distribution))       # all loads

# Storage parameters
Pch, Pdis, Emax = 150.0, 150.0, 10000.0  # charge/discharge power limits (MW) and energy capacity (MWh)
eta_ch, eta_dis = 0.90, 0.95 # charge/discharge efficiency
E0 = 0 * Emax  # initial state of charge (50%)

# Build generator capacities and hourly demand
gen_capacity = build_hourly_gen_capacity(conventional_generators, wind_capacity)
hourly_demand_data = build_hourly_demand_data(load_nodes, load_percentages, elastic_nodes, elastic_bid_prices, loads)



# Variables
VARIABLES = build_variables(GENERATORS, LOADS, T)

# Objective
objective_coeff = build_objective_coeff(VARIABLES, GENERATORS, LOADS, T, total_generators['cost'].values, hourly_demand_data)

# Constraints
CONSTRAINTS, constraints_coeff, constraints_rhs, constraints_sense = build_constraints(
    GENERATORS, LOADS, T, gen_capacity, hourly_demand_data, Pch, Pdis, Emax, E0, eta_ch, eta_dis
)

# LP input data
input_data = LP_InputData(
    VARIABLES=VARIABLES,
    CONSTRAINTS=CONSTRAINTS,
    objective_coeff=objective_coeff,
    constraints_coeff=constraints_coeff,
    constraints_rhs=constraints_rhs,
    constraints_sense=constraints_sense,
    objective_sense=GRB.MAXIMIZE,
    model_name="24h Market Clearing with Storage"
)

# Solve
model = LP_OptimizationProblem(input_data)
model.run()



# 1. Extract results for each hour
results_by_hour = []
for t in range(24):
    res = extract_hourly_results(model, GENERATORS, LOADS, load_nodes, elastic_nodes, hourly_demand_data, total_generators, t)
    results_by_hour.append(res)

results_df = pd.DataFrame(results_by_hour)

# 2. Plot 24-hour results
#plot_24h_results(results_df)

# 3. Plot merit order for a chosen hour
#plot_merit_order(load_nodes, elastic_nodes, elastic_bid_prices, loads, hour, results_df, conventional_generators, wind_generator, wind_capacity)
  
print(results_df['battery_charge'])