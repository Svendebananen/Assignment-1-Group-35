# step 1: single hour optimization with 6/17 elastic loads
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
os.chdir(Path(__file__).parent)

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

def LP_builder(
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],              
        constraints_coeff: dict[str, dict[str,float]],  
        constraints_rhs: dict[str, float],              
        constraints_sense: dict[str, int],              
        objective_sense: int,                           
        model_name: str                                 
): 
    # Build model
    model = gp.Model(name=model_name)

    # add variables
    variables = {v: model.addVar(lb=0, name=f'{v}') for v in VARIABLES}

    # Objective
    objective = gp.quicksum(objective_coeff[v] * variables[v] for v in VARIABLES)
    model.setObjective(objective, objective_sense)

    # Constraints
    for c in CONSTRAINTS:
        model.addLConstr(gp.quicksum(constraints_coeff[c][v] * variables[v] for v in VARIABLES), constraints_sense[c], constraints_rhs[c], name=f'{c}')
    model.update()
    return model


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


# Creating a DataFrame to concatenate later
# Hour selected for merit order curve analysis (0-based index)
hour = 4  # hour 5

# Optimization for each hour
# Define ranges and indexes
N_GENERATORS = len(total_generators) # number of generators (12 conventional + 6 wind)
N_LOADS = len(load_distribution) # number of loads (17 total: 11 inelastic + 6 elastic)
time_step = 24 # time step in hours 
GENERATORS = range(len(total_generators)) 
LOADS = range(N_LOADS) 

# Storage for results across all hours
results_by_hour = []
for t in range(time_step):  # Loop over time steps (hours)
    wind_generator['capacity'] = wind_capacity[:, t] # Update wind generator capacities for the current hour based on CSV data
    total_generators =  pd.concat([conventional_generators, wind_generator], ignore_index=True) # Update total generators DataFrame with the new wind generator capacities for the current hour
    
    total_demand = loads['demand'][t] # update total demand for the current hour
    print(total_demand)
    
    
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
    
    total_generation = sum(model.results.variables[f'production of generator {g}'] for g in GENERATORS) #should always be the same  to demand as we set constraint to equality
    total_demand_served = sum(model.results.variables[f'demand of load {j}'] for j in LOADS)
    
    # Calculate elastic vs inelastic served
    elastic_demand_base = sum(total_demand * load_percentages[node] for node in elastic_nodes)
    inelastic_demand_base = sum(total_demand * load_percentages[node] for node in load_nodes if node not in elastic_nodes)
    
    elastic_served = sum(model.results.variables[f'demand of load {j}'] for j, node in enumerate(load_nodes) if node in elastic_nodes)
    inelastic_served = sum(model.results.variables[f'demand of load {j}'] for j, node in enumerate(load_nodes) if node not in elastic_nodes)
    
    total_cost = sum(total_generators['cost'][g] * model.results.variables[f'production of generator {g}'] for g in GENERATORS)
    
    social_welfare = model.results.objective_value
    
    consumer_utility = sum(demand_data['bid_price'][j] * model.results.variables[f'demand of load {j}'] for j in LOADS)
    
    producer_surplus = mcp * total_generation - total_cost
    
    # Calculate flexibility index 
    # Negative = curtailed (below base), Positive = increased (above base)
    elastic_flexibility_pct = (elastic_served / elastic_demand_base - 1) * 100 if elastic_demand_base > 0 else 0
    
    # Also calculate curtailment from MAX (110%) for reference
    elastic_max_possible = elastic_demand_base * 1.10
    elastic_curtailment_from_max = (1 - elastic_served / elastic_max_possible) * 100 if elastic_max_possible > 0 else 0
    if t == 4:  # Store detailed results for the hour selected for merit order curve analysis
        producer_profits = {}
        utility_by_load = {}

        for g in GENERATORS:
            p = model.results.variables[f'production of generator {g}']
            cost = total_generators['cost'][g]
            producer_profits[g] = mcp * p - cost * p

        for j in LOADS:
            q = model.results.variables[f'demand of load {j}']
            bid_price = demand_data['bid_price'][j]
            utility_by_load[j] = (bid_price-mcp) * q

    # ========== STORE RESULTS ==========
    results_by_hour.append({
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
        'producer_surplus': producer_surplus
    })
    
results_df = pd.DataFrame(results_by_hour)


# Print summary for the selected hour
h = results_df[results_df['hour'] == hour + 1].iloc[0]
print(f'\n--- Summary for Hour {hour + 1} ---')
print(f'Market Clearing Price: €{h["mcp"]:.2f}/MWh')
print(f'Total Generation: {h["generation"]:.2f} MW')
print(f'Total Demand Served: {h["demand_total"]:.2f} MW')
print(f'  - Inelastic: {h["demand_inelastic"]:.2f} MW (base: {h["demand_inelastic"]:.2f} MW)')
print(f'  - Elastic: {h["demand_elastic"]:.2f} MW (base: {h["elastic_base"]:.2f} MW)')
print(f'  - Elastic Flexibility: {h["elastic_flexibility_pct"]:+.1f}%')
print(f'Social Welfare: €{h["social_welfare"]:.2f}')
print(f'Total Cost: €{h["total_cost"]:.2f}')
print(f'Producer Surplus: €{h["producer_surplus"]:.2f}')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Graph 1: Market Clearing Price
axes[0, 0].plot(results_df['hour'], results_df['mcp'], marker='o', linewidth=2, color='blue')
axes[0, 0].set_xlabel('Hour', fontweight='bold')
axes[0, 0].set_ylabel('Market Clearing Price (€/MWh)', fontweight='bold')
axes[0, 0].set_title('Market Clearing Price - 24 Hours', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(1, 25))

# Graph 2: Social Welfare
axes[0, 1].plot(results_df['hour'], results_df['social_welfare'], marker='s', linewidth=2, color='green')
axes[0, 1].set_xlabel('Hour', fontweight='bold')
axes[0, 1].set_ylabel('Social Welfare (€)', fontweight='bold')
axes[0, 1].set_title('Social Welfare - 24 Hours', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(1, 25))

# Graph 3: Demand Breakdown
axes[1, 0].plot(results_df['hour'], results_df['demand_inelastic'], marker='^', linewidth=2, 
                color='orange', label='Inelastic Demand')
axes[1, 0].plot(results_df['hour'], results_df['demand_elastic'], marker='v', linewidth=2, 
                color='purple', label='Elastic Demand')
axes[1, 0].plot(results_df['hour'], results_df['demand_total'], marker='o', linewidth=2, 
                color='red', label='Total Demand', linestyle='--')
axes[1, 0].set_xlabel('Hour', fontweight='bold')
axes[1, 0].set_ylabel('Demand (MW)', fontweight='bold')
axes[1, 0].set_title('Demand Components - 24 Hours', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(1, 25))

# Graph 4: Elastic Flexibility
flex_values = results_df['elastic_flexibility_pct'].values
colors = ['green' if x > 0 else 'red' for x in flex_values]

axes[1, 1].bar(results_df['hour'], flex_values, color=colors, alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.0)
axes[1, 1].set_xlabel('Hour', fontweight='bold')
axes[1, 1].set_ylabel('Elastic Flexibility (%)', fontweight='bold')
axes[1, 1].set_title('Elastic Demand Flexibility - 24 Hours\n(Negative=Curtailed, Positive=Increased)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_xticks(range(1, 25))

plt.tight_layout()
plt.show()

# ============================================================================
# MERIT ORDER CURVE + DEMAND CURVE (chosen hour)
# ============================================================================

# --- Supply curve ---
moc_total_demand = loads['demand'][hour]
wind_generator['capacity'] = wind_capacity[:, hour]
moc_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)
moc_generators_sorted = moc_generators.copy().sort_values(by=["cost"])

supply_cumulative = []
supply_cost = []
for i in range(len(moc_generators_sorted)):
    supply_cumulative.append(sum(moc_generators_sorted['capacity'][:i]))
    supply_cost.append(moc_generators_sorted['cost'].iloc[i])

# --- Demand curve ---
# Build list of (quantity, bid_price) for each load at the selected hour
demand_bids = []
for node in load_nodes:
    qty = moc_total_demand * load_percentages[node]
    if node in elastic_nodes:
        bid = elastic_bid_prices[node]
        demand_bids.append((qty * 1.10, bid))   # elastic: max quantity
    else:
        demand_bids.append((qty, 300.0))         # inelastic: fixed quantity

# Sort by bid price descending (highest willingness to pay first)
demand_bids.sort(key=lambda x: x[1], reverse=True)

demand_cumulative = [0]
demand_prices = []
for qty, bid in demand_bids:
    demand_cumulative.append(demand_cumulative[-1] + qty)
    demand_prices.append(bid)

# MCP from optimiser
moc_equilibrium = results_df.loc[results_df['hour'] == hour + 1, 'mcp'].values[0]

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 7))

# Supply curve
ax.step(supply_cumulative, supply_cost, where='post', linewidth=2.5, color='steelblue', label='Supply Curve')
ax.fill_between(supply_cumulative, supply_cost, step='post', alpha=0.2, color='steelblue')

# Demand curve (step descending)
demand_x = []
demand_y = []
for i, (qty, bid) in enumerate(demand_bids):
    demand_x.append(demand_cumulative[i])
    demand_y.append(bid)
    demand_x.append(demand_cumulative[i + 1])
    demand_y.append(bid)

# Final point: drop to zero
demand_x.append(demand_cumulative[-1])
demand_y.append(0)

ax.plot(demand_x, demand_y, linewidth=2.5, color='red', label='Demand Curve')

# MCP line
ax.axhline(y=moc_equilibrium, color='green', linestyle='--', linewidth=2, label=f'MCP: €{moc_equilibrium:.2f}/MWh')

ax.set_xlabel('Cumulative Capacity / Demand (MW)', fontsize=12, fontweight='bold')
ax.set_ylabel('Price (€/MWh)', fontsize=12, fontweight='bold')
ax.set_title(f'Merit Order & Demand Curve - Hour {hour + 1}', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("\n" + "24-HOUR MARKET CLEARING SUMMARY")
print(f"Average Market Clearing Price: €{results_df['mcp'].mean():.2f}/MWh")
print(f"Peak MCP: €{results_df['mcp'].max():.2f}/MWh (Hour {results_df.loc[results_df['mcp'].idxmax(), 'hour']:.0f})")
print(f"Minimum MCP: €{results_df['mcp'].min():.2f}/MWh (Hour {results_df.loc[results_df['mcp'].idxmin(), 'hour']:.0f})")

print(f"\nAverage Elastic Flexibility: {results_df['elastic_flexibility_pct'].mean():+.1f}%")
print(f"Max Elastic Increase: {results_df['elastic_flexibility_pct'].max():+.1f}% (Hour {results_df.loc[results_df['elastic_flexibility_pct'].idxmax(), 'hour']:.0f})")
print(f"Max Elastic Curtailment: {results_df['elastic_flexibility_pct'].min():+.1f}% (Hour {results_df.loc[results_df['elastic_flexibility_pct'].idxmin(), 'hour']:.0f})")

print(f"\nTotal Social Welfare (24h): €{results_df['social_welfare'].sum():.2f}")
print(f"Total Generation Cost (24h): €{results_df['total_cost'].sum():.2f}")
print(f"Total Producer Surplus (24h): €{results_df['producer_surplus'].sum():.2f}")
print(f"Total Consumer Utility (24h): €{results_df['consumer_utility'].sum():.2f}")

print(producer_profits)
print(utility_by_load)
     
