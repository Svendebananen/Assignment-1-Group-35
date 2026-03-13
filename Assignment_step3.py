# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together! 

# step 3: single hour optimization with 6/17 elastic loads and tranmission line constraints

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
# importing os library to consider the same folder of this file for the csv reading
import os
from types import SimpleNamespace
from pathlib import Path
os.chdir(Path(__file__).parent)

# mute the gurobi license print
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start() 

Expando = SimpleNamespace
data = Expando()

#define classes for input data and optimization problem
class LP_InputData:

    def __init__(
        self, 
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],               # Coefficients in objective function
        constraints_coeff: dict[str, dict[str,float]],    # Linear coefficients of constraints
        constraints_rhs: dict[str, float],                # Right hand side coefficients of constraints
        constraints_sense: dict[str, str],              # Direction of constraints
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
        self.variables = {}
        for v in self.data.VARIABLES:
            # Voltage angles and power flows can be negative
            if 'voltage angle' in v or 'power flow' in v:
                self.variables[v] = self.model.addVar(lb=-GRB.INFINITY, name=f'{v}')
            else:
                self.variables[v] = self.model.addVar(lb=0, name=f'{v}')
    
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
  wind_capacity[i,:] = data['capacity_factor'][index:index+24].values * 200 

wind_generator = pd.DataFrame({ # wind generators data, with capacity to be updated for each hour based on CSV files
        'id': [f'wind_{i}' for i in range(wind_capacity.shape[0])],
        'bus': pd.read_csv('wind_farms.csv',usecols=['node'])['node'].values,
        'capacity': 0.0,# placeholder, will be updated for each hour
        'cost': [0.0 for i in range(wind_capacity.shape[0])]
    }) 

# Creating a single DataFrame with all generators (conventional + wind)
total_generators =  pd.concat([conventional_generators, wind_generator], ignore_index = True)

# Transmission lines data
lines = pd.read_csv('transmission_lines.csv')



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
elastic_bid_price = df.set_index('node')['bid'].to_dict()

# Hour selected for merit order curve analysis (0-based index)
hour = 4  # hour 5

# Optimization for each hour
# Define ranges and indexes
N_GENERATORS = len(total_generators) # number of generators (12 conventional + 6 wind)
N_LOADS = len(load_distribution) # number of loads (17 total: 11 inelastic + 6 elastic)
time_step = 24 # time step in hours 
GENERATORS = range(len(total_generators)) 
LOADS = range(N_LOADS) 
LINES = range(len(lines))
# All unique nodes: combine nodes from loads, generators, and transmission lines
all_nodes = set(int(n) for n in load_nodes) | \
            set(int(n) for n in total_generators['bus'].values) | \
            set(int(n) for n in lines['from_node'].values) | set(int(n) for n in lines['to_node'].values)
NODES = sorted(all_nodes)  # All unique nodes in the system

# Storage for results across all hours
results_by_hour = []
lmp_rows = []
for t in range(time_step):  # Loop over time steps (hours)
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
            bid_prices.append(elastic_bid_price[node])
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
    
    
    # Build dictionaries to map nodes to generators and loads
    generators_at_node = {n: [] for n in NODES}
    for g in GENERATORS:
        node = total_generators['bus'][g]
        generators_at_node[node].append(g)
    
    loads_at_node = {n: [] for n in NODES}
    for j in LOADS:
        node = demand_data['node'][j]
        loads_at_node[node].append(j)
    
    # Create dictionaries for lines entering and leaving each node
    lines_from_node = {n: [] for n in NODES}
    lines_to_node = {n: [] for n in NODES}
    for i in LINES:
        from_n = lines['from_node'][i]
        to_n = lines['to_node'][i]
        lines_from_node[from_n].append(i)
        lines_to_node[to_n].append(i)
    
    input_data = {
        'model0': LP_InputData(
            VARIABLES = [f'production of generator {g}' for g in GENERATORS] + \
                        [f'demand of load {j}' for j in LOADS] + \
                        [f'voltage angle at node {n}' for n in NODES] + \
                        [f'power flow line {i}' for i in LINES],

            CONSTRAINTS = [f'balance constraint node {n}' for n in NODES] + \
                          [f'capacity constraint {g}' for g in GENERATORS] + \
                          [f'demand min limit {j}' for j in LOADS] + \
                          [f'demand max limit {j}' for j in LOADS] + \
                          [f'flow definition line {i}' for i in LINES] + \
                          [f'transmission limit positive {i}' for i in LINES] + \
                          [f'transmission limit negative {i}' for i in LINES] + \
                          ['angle reference node constraint'],
            
            objective_coeff = {
                # Consumer utility (positive)
                **{f'demand of load {j}': demand_data['bid_price'][j] for j in LOADS},
                # Generation cost (negative)
                **{f'production of generator {g}': -total_generators['cost'][g] for g in GENERATORS},
                # Voltage angles and power flows don't affect objective
                **{f'voltage angle at node {n}': 0 for n in NODES},
                **{f'power flow line {i}': 0 for i in LINES}
            },
            
            constraints_coeff = {
                # Nodal balance: Generation - Demand - Outflow + Inflow = 0
                **{f'balance constraint node {n}': {
                    # Generation at this node
                    **{f'production of generator {g}': 1 for g in generators_at_node[n]},
                    # Demand at this node  
                    **{f'demand of load {j}': -1 for j in loads_at_node[n]},
                    # Outflows (lines leaving this node)
                    **{f'power flow line {i}': -1 for i in lines_from_node[n]},
                    # Inflows (lines entering this node)
                    **{f'power flow line {i}': 1 for i in lines_to_node[n]}
                } for n in NODES},
                
                # Flow definition: P_ij = (θ_i - θ_j) / X_ij
                # Rearranged: P_ij * X_ij - θ_i + θ_j = 0
                **{f'flow definition line {i}': {
                    f'power flow line {i}': lines['reactance_pu'][i],
                    f'voltage angle at node {lines["from_node"][i]}': -1,
                    f'voltage angle at node {lines["to_node"][i]}': 1
                } for i in LINES},
                
                # Generator capacity
                **{f'capacity constraint {g}': {f'production of generator {k}': int(k == g) for k in GENERATORS} for g in GENERATORS},
                
                # Demand minimum limits
                **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
                
                # Demand maximum limits
                **{f'demand max limit {j}': {f'demand of load {j}': 1} for j in LOADS},
                
                # Transmission capacity constraints: -Cap <= P_ij <= Cap
                **{f'transmission limit positive {i}': {f'power flow line {i}': 1} for i in LINES},
                **{f'transmission limit negative {i}': {f'power flow line {i}': 1} for i in LINES},
                
                # Reference node angle constraint (slack bus at node 13)
                'angle reference node constraint': {f'voltage angle at node 13': 1}
            },
            
            constraints_rhs = {
                **{f'balance constraint node {n}': 0 for n in NODES},
                **{f'flow definition line {i}': 0 for i in LINES},
                **{f'capacity constraint {g}': total_generators['capacity'][g] for g in GENERATORS},
                **{f'demand min limit {j}': demand_data['bid_quantity_min'][j] for j in LOADS},
                **{f'demand max limit {j}': demand_data['bid_quantity_max'][j] for j in LOADS},
                **{f'transmission limit positive {i}': lines['capacity_MVA'][i] for i in LINES},
                **{f'transmission limit negative {i}': -lines['capacity_MVA'][i] for i in LINES},
                'angle reference node constraint': 0
            },
            
            constraints_sense = { 
                **{f'balance constraint node {n}': GRB.EQUAL for n in NODES},
                **{f'flow definition line {i}': GRB.EQUAL for i in LINES},
                **{f'capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS},
                **{f'demand min limit {j}': GRB.GREATER_EQUAL for j in LOADS},
                **{f'demand max limit {j}': GRB.LESS_EQUAL for j in LOADS},
                **{f'transmission limit positive {i}': GRB.LESS_EQUAL for i in LINES},
                **{f'transmission limit negative {i}': GRB.GREATER_EQUAL for i in LINES},
                'angle reference node constraint': GRB.EQUAL
            },
            
            objective_sense = GRB.MAXIMIZE, # maximize social welfare
            model_name = "Market Clearing - 17 Loads (11 Inelastic + 6 Elastic) - with Transmission Constraints"
     )
    }
    
    model = LP_OptimizationProblem(input_data['model0'])
    model.run()
    
    # Nodal LMPs are the negatives of nodal-balance duals with this sign convention.
    lmp_by_node = {
        n: -model.results.optimal_duals[f'balance constraint node {n}']
        for n in NODES
    }

    # Market clearing price at reference node (node 13)
    mcp = lmp_by_node[13]

    for n in NODES:
        lmp_rows.append({'hour': t + 1, 'node': n, 'lmp': lmp_by_node[n]})
    
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
    
    producer_surplus = sum(
    lmp_by_node[total_generators['bus'].iloc[g]] * model.results.variables[f'production of generator {g}'] 
    - total_generators['cost'].iloc[g] * model.results.variables[f'production of generator {g}']
    for g in GENERATORS)
    
    # Calculate flexibility index 
    # Negative = curtailed (below base), Positive = increased (above base)
    elastic_flexibility_pct = (elastic_served / elastic_demand_base - 1) * 100 if elastic_demand_base > 0 else 0
    
    # Also calculate curtailment from MAX (110%) for reference
    elastic_max_possible = elastic_demand_base * 1.10
    elastic_curtailment_from_max = (1 - elastic_served / elastic_max_possible) * 100 if elastic_max_possible > 0 else 0
    if t == hour:  # Store detailed results for the selected hour
        model_selected = model
        generators_selected = total_generators.copy()
        lmp_by_node_selected = lmp_by_node.copy()
        producer_profits = {}
        utility_by_load = {}

        for g in GENERATORS:
            p = model.results.variables[f'production of generator {g}']
            cost = total_generators['cost'][g]
            gen_node = total_generators['bus'].iloc[g]
            producer_profits[g] = (lmp_by_node[gen_node] - cost) * p    #nodal market means Gen gets paid for the LMP at the node

        for j in LOADS:
            q = model.results.variables[f'demand of load {j}']
            bid_price = demand_data['bid_price'][j]
            load_node = demand_data['node'].iloc[j]
            utility_by_load[j] = (bid_price - lmp_by_node[load_node]) * q   #same as before, costumer pays the LMP at the node

    # Store results
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
print('\n'f'Step 3 market-clearing outcomes for {hour + 1}:') 
print(f'Market Clearing Price: €{h["mcp"]:.2f}/MWh') 
print(f'Total Operatring Cost: €{h["total_cost"]:.2f}')
print(f'Social Welfare: €{h["social_welfare"]:.2f}')
print('\n')
print("Locational Marginal Price (LMP) by node:")
print(f"{'Node':<10} {'LMP (€/MWh)':<15}")
for n in NODES:
    print(f"{n:<10} {lmp_by_node_selected[n]:<15.4f}")
print('\n')
print(f"{'Node':<10}  {'Utility':<10}")
for j in LOADS:
    print(f"{f'{j+1}':<10} {utility_by_load[j]:<10.2f}") 
print('\n')
print(f"{'Generator':<10}  {'Profit':<10}")
for g in GENERATORS:
    print(f"{f'{g+1}':<10} {producer_profits[g]:<10.2f}") 
print('\n') 
print("Verifiy the market-clearing price using the KKT conditions")
lam = -model_selected.results.optimal_duals['balance constraint node 13']
print(f"MCP (λ) at reference node 13: {lam:.4f}")
# Stationarity
for g in GENERATORS:
    mu = model_selected.results.optimal_duals[f'capacity constraint {g}']
    cost = generators_selected['cost'].iloc[g]  # ← corretto
    print(f"Gen {g+1}: cost={cost:.2f}, μ={mu:.4f}, λ - C - μ = {lam - cost - mu:.6f}")

# Complementary slackness
for g in GENERATORS:
    p = model_selected.results.variables[f'production of generator {g}']
    cap = generators_selected['capacity'].iloc[g]  # ← corretto
    mu = model_selected.results.optimal_duals[f'capacity constraint {g}']
    print(f"Gen {g+1}: μ*(P_max - p) = {mu * (cap - p):.6f}")

print('\nExtra info:')
print(f'Total Generation: {h["generation"]:.2f} MW')
print(f'Total Demand Served: {h["demand_total"]:.2f} MW')
print(f'  - Inelastic: {h["demand_inelastic"]:.2f} MW (base: {h["demand_inelastic"]:.2f} MW)')
print(f'  - Elastic: {h["demand_elastic"]:.2f} MW (base: {h["elastic_base"]:.2f} MW)')
print(f'  - Elastic Flexibility: {h["elastic_flexibility_pct"]:+.1f}%')


print(f'Producer Surplus: €{h["producer_surplus"]:.2f}')

# Transmission diagnostics for selected hour
flow_values = {i: model_selected.results.variables[f'power flow line {i}'] for i in LINES}
line_caps = {i: float(lines['capacity_MVA'][i]) for i in LINES}
line_utilization = {i: abs(flow_values[i]) / line_caps[i] if line_caps[i] > 0 else 0 for i in LINES}
binding_lines = [i for i in LINES if abs(abs(flow_values[i]) - line_caps[i]) <= 1e-4]

print('\nTransmission diagnostics (selected hour):')
print(f'Max line utilization: {max(line_utilization.values()) * 100:.2f}%')
print(f'Number of binding line limits: {len(binding_lines)} out of {len(list(LINES))}')
if binding_lines:
    print('Binding lines (1-based index):', [i + 1 for i in binding_lines])

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

# Graph 3: Demand Breakdown (elastic vs inelastic)
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

# Supply curve
moc_total_demand = loads['demand'][hour]
wind_generator['capacity'] = wind_capacity[:, hour]
moc_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)
moc_generators_sorted = moc_generators.copy().sort_values(by=["cost"])

supply_cumulative = []
supply_cost = []
for i in range(len(moc_generators_sorted)):
    supply_cumulative.append(sum(moc_generators_sorted['capacity'][:i]))
    supply_cost.append(moc_generators_sorted['cost'].iloc[i])

# Demand curve
# Build list of (quantity, bid_price) for each load at the selected hour
demand_bids = []
for node in load_nodes:
    qty = moc_total_demand * load_percentages[node]
    if node in elastic_nodes:
        bid = elastic_bid_price[node]
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

# Plot
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
