# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together! 

# step 2: multi-hour optimization with battery storage
# imports
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from pathlib import Path
os.chdir(Path(__file__).parent)

# mute the gurobi license print
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()  
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Expando(object):
    pass

#define classes for input data and optimization problem
class LP_InputData:
    def __init__(
        self,
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],
        constraints_coeff: dict[str, dict[str, float]],
        constraints_rhs: dict[str, float],
        constraints_sense: dict[str, int],
        objective_sense: int,
        model_name: str
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
    def __init__(self, input_data: LP_InputData):
        self.data = input_data
        self.results = Expando()
        self._build_model() 


    def _build_variables(self):
        self.variables = {v: self.model.addVar(lb=0, name=f'{v}') for v in self.data.VARIABLES}

    def _build_constraints(self):
        self.constraints = {c:
            self.model.addLConstr(
                gp.quicksum(self.data.constraints_coeff[c].get(v, 0) * self.variables[v] for v in self.data.VARIABLES),
                self.data.constraints_sense[c],
                self.data.constraints_rhs[c],
                name=f'{c}'
            ) for c in self.data.CONSTRAINTS
        }

    def _build_objective_function(self):
        objective = gp.quicksum(self.data.objective_coeff[v] * self.variables[v] for v in self.data.VARIABLES)
        self.model.setObjective(objective, self.data.objective_sense)

    def _build_model(self):
        self.model = gp.Model(name=self.data.model_name, env=env)
        self.model.setParam('OutputFlag', 0)
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.variables = {v.VarName: v.x for v in self.model.getVars()}
        self.results.optimal_duals = {f'{c.ConstrName}': c.Pi for c in self.model.getConstrs()}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {self.model.ModelName} was not successful")


def build_multi_hour_input_data(
    generator_cost,
    generator_capacity,
    load_capacity,
    generators_range,
    load_nodes,
    load_percentages,
    elastic_nodes,
    elastic_bid_prices,
    peak_multiplier,
    time_range,
    wind_capacity_by_hour=None,
):
    # Generator variables: production at each hour
    variables = [f'production of generator {g} at hour {t}' for t in time_range for g in generators_range]

    # Battery variables: charge power, discharge power, and state of charge
    variables += [f'battery charge at hour {t}' for t in time_range]
    variables += [f'battery discharge at hour {t}' for t in time_range]
    variables += [f'battery SOC at hour {t}' for t in time_range]

    # Demand variables: accepted quantity for each load at each hour
    variables += [f'demand of load {j} at hour {t}' for t in time_range for j in range(len(load_nodes))]

    # Balance constraints
    constraints = [f'balance constraint at hour {t}' for t in time_range]

    # Generator capacity constraints
    constraints += [f'capacity constraint {g} at hour {t}' for t in time_range for g in generators_range]

    # Demand min/max constraints
    constraints += [f'demand min limit {j} at hour {t}' for t in time_range for j in range(len(load_nodes))]
    constraints += [f'demand max limit {j} at hour {t}' for t in time_range for j in range(len(load_nodes))]

    # Battery constraints
    constraints += [f'battery charge limit at hour {t}' for t in time_range]
    constraints += [f'battery discharge limit at hour {t}' for t in time_range]
    constraints += [f'battery SOC limit at hour {t}' for t in time_range]
    constraints += [f'battery dynamics at hour {t}' for t in time_range]

    # Objective: maximize social welfare = consumer utility - generation cost
    # Generation cost (negative)
    objective_coeff = {
        f'production of generator {g} at hour {t}': -generator_cost[g]
        for t in time_range for g in generators_range
    }
    # Battery at zero cost
    for t in time_range:
        objective_coeff[f'battery charge at hour {t}'] = 0
        objective_coeff[f'battery discharge at hour {t}'] = 0
        objective_coeff[f'battery SOC at hour {t}'] = 0
    # Consumer utility (positive)
    for t in time_range:
        for j, node in enumerate(load_nodes):
            if node in elastic_nodes:
                bid_price = elastic_bid_prices[node] * peak_multiplier[t]
            else:
                bid_price = 300.0
            objective_coeff[f'demand of load {j} at hour {t}'] = bid_price

    constraints_coeff = {}
    constraints_rhs = {}
    constraints_sense = {}

    for t in time_range:
        # Balance constraint: demand - generation - discharge + charge = 0
        balance_name = f'balance constraint at hour {t}'
        constraints_coeff[balance_name] = {
            **{f'demand of load {j} at hour {t}': 1 for j in range(len(load_nodes))},
            **{f'production of generator {g} at hour {t}': -1 for g in generators_range},
            f'battery discharge at hour {t}': -1,
            f'battery charge at hour {t}': 1,
        }
        constraints_rhs[balance_name] = 0
        constraints_sense[balance_name] = GRB.EQUAL

        # Generator capacity constraints
        for g in generators_range:
            cap_name = f'capacity constraint {g} at hour {t}'
            constraints_coeff[cap_name] = {f'production of generator {g} at hour {t}': 1}
            if wind_capacity_by_hour is not None and g >= 12:
                wind_idx = g - 12
                constraints_rhs[cap_name] = wind_capacity_by_hour[wind_idx, t]
            else:
                constraints_rhs[cap_name] = generator_capacity[g]
            constraints_sense[cap_name] = GRB.LESS_EQUAL

        # Demand min/max constraints
        for j, node in enumerate(load_nodes):
            demand_at_node = load_capacity[t] * load_percentages[node]
            if node in elastic_nodes:
                min_qty = demand_at_node * 0.20
                max_qty = demand_at_node * 1.10
            else:
                min_qty = demand_at_node
                max_qty = demand_at_node

            constraints_coeff[f'demand min limit {j} at hour {t}'] = {f'demand of load {j} at hour {t}': 1}
            constraints_rhs[f'demand min limit {j} at hour {t}'] = min_qty
            constraints_sense[f'demand min limit {j} at hour {t}'] = GRB.GREATER_EQUAL

            constraints_coeff[f'demand max limit {j} at hour {t}'] = {f'demand of load {j} at hour {t}': 1}
            constraints_rhs[f'demand max limit {j} at hour {t}'] = max_qty
            constraints_sense[f'demand max limit {j} at hour {t}'] = GRB.LESS_EQUAL

        # Battery charge limit
        constraints_coeff[f'battery charge limit at hour {t}'] = {f'battery charge at hour {t}': 1}
        constraints_rhs[f'battery charge limit at hour {t}'] = BATTERY_POWER_MAX_CHARGE
        constraints_sense[f'battery charge limit at hour {t}'] = GRB.LESS_EQUAL

        # Battery discharge limit
        constraints_coeff[f'battery discharge limit at hour {t}'] = {f'battery discharge at hour {t}': 1}
        constraints_rhs[f'battery discharge limit at hour {t}'] = BATTERY_POWER_MAX_DISCHARGE
        constraints_sense[f'battery discharge limit at hour {t}'] = GRB.LESS_EQUAL

        # Battery SOC limit
        constraints_coeff[f'battery SOC limit at hour {t}'] = {f'battery SOC at hour {t}': 1}
        constraints_rhs[f'battery SOC limit at hour {t}'] = BATTERY_ENERGY_MAX
        constraints_sense[f'battery SOC limit at hour {t}'] = GRB.LESS_EQUAL

        # Battery dynamics: e_t = e_{t-1} + eta_ch * p_ch_t - p_dis_t / eta_dis
        batt_dyn = f'battery dynamics at hour {t}'
        constraints_coeff[batt_dyn] = {
            f'battery SOC at hour {t}': 1,
            f'battery charge at hour {t}': -BATTERY_ETA_CHARGE,
            f'battery discharge at hour {t}': 1 / BATTERY_ETA_DISCHARGE
        }
        if t > 0:
            constraints_coeff[batt_dyn][f'battery SOC at hour {t-1}'] = -1
            constraints_rhs[batt_dyn] = 0
        else:
            constraints_rhs[batt_dyn] = BATTERY_INITIAL_SOC
        constraints_sense[batt_dyn] = GRB.EQUAL

    return LP_InputData(
        VARIABLES=variables,
        CONSTRAINTS=constraints,
        objective_coeff=objective_coeff,
        constraints_coeff=constraints_coeff,
        constraints_rhs=constraints_rhs,
        constraints_sense=constraints_sense,
        objective_sense=GRB.MAXIMIZE,
        model_name="Market Clearing multi-hour with battery",
    )

# Import data from case study
date = '2019-08-31' # Choose data for wind turbine generation

conventional_generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id', 'bus', 'capacity', 'cost']) # conventional generators data

# Creating the wind capacity matrix for 6 wind generators and 24 hours
wind_capacity = np.zeros((6, 24))
file_list = glob.glob('Ninja/*.csv')
for i, csv in enumerate(file_list):
    data = pd.read_csv(csv, header=None, names=['time', 'local_time', 'capacity_factor'], skiprows=4)
    index = data.loc[data['time'] == date + ' 00:00'].index[0]
    wind_capacity[i, :] = data['capacity_factor'][index:index + 24].values * 200

# Creating a DataFrame for wind farms, to concatenate later with the conventional generators DataFrame
wind_df = pd.DataFrame({
    'id': [f'wind_{i}' for i in range(6)],
    'bus': pd.read_csv('wind_farms.csv',usecols=['node'])['node'].values,
    'capacity': [200.0] * 6,  # placeholder, overridden by wind_capacity_by_hour
    'cost': [0.0] * 6
})

# Combine conventional and wind generators
generators_combined = pd.concat([conventional_generators, wind_df], ignore_index=True)

# Load demand data
loads = pd.read_csv('LoadData.csv', header=None, names=['hour', 'demand'])
load_distribution = pd.read_csv('load_distribution_1.csv')
load_nodes = load_distribution['node'].tolist()
load_percentages = dict(zip(load_distribution['node'], load_distribution['pct_of_system_load'] / 100))

# List of elastic loads: nodes 1, 7, 9, 13, 14, 15
elastic_nodes = [1, 7, 9, 13, 14, 15] 

# Bid prices for elastic loads
elastic_bid_prices = {1: 12.0, 7: 22.0, 9: 10.0, 13: 20.0, 14: 16.0, 15: 25.0} 

# demand bid quantities and bid prices must vary across hours (comparatively higher during peak hours)
peak_multiplier = {t: 1.3 if t in range(7, 10) or t in range(16, 20) else 1.0 for t in range(24)}

# Define ranges and indexes
time_step = 24
GENERATORS = range(len(generators_combined))        # 18: 12 conventional + 6 wind
WINDTURBINES = range(12, len(generators_combined))  # 6 wind farms
LOADS = range(len(load_distribution))               # 17: 11 inelastic + 6 elastic
N_GENERATORS = len(GENERATORS)
N_LOADS = len(LOADS)

# Battery parameters
BATTERY_ENERGY_MAX = 100.0
BATTERY_POWER_MAX_DISCHARGE = 50.0
BATTERY_POWER_MAX_CHARGE = 50.0
BATTERY_ETA_CHARGE = 0.93
BATTERY_ETA_DISCHARGE = 0.95
BATTERY_INITIAL_SOC = 0.0

generator_cost = generators_combined['cost']
load_capacity = loads['demand']

# build and solve model
multi_hour_data = build_multi_hour_input_data(
    generator_cost = generator_cost,
    generator_capacity = generators_combined['capacity'],
    load_capacity = load_capacity,
    generators_range = GENERATORS,
    load_nodes = load_nodes,
    load_percentages = load_percentages,
    elastic_nodes = elastic_nodes,
    elastic_bid_prices = elastic_bid_prices,
    peak_multiplier = peak_multiplier,
    time_range = range(time_step),
    wind_capacity_by_hour = wind_capacity,
)

multi_hour_model = LP_OptimizationProblem(multi_hour_data)
multi_hour_model.run()

# Print results
print(f"Total Social Welfare: €{multi_hour_model.results.objective_value:.2f}") 

total_operating_cost = 0
for t in range(time_step):
    for g in GENERATORS:
        production = multi_hour_model.results.variables[f'production of generator {g} at hour {t}']
        cost = generators_combined['cost'].iloc[g]
        total_operating_cost += production * cost 
print(f'Total Operating Cost: €{total_operating_cost:.2f}') 

storage_profit = 0
for t in range(time_step):
    discharge = multi_hour_model.results.variables.get(f'battery discharge at hour {t}', 0)
    charge = multi_hour_model.results.variables.get(f'battery charge at hour {t}', 0)
    mcp = multi_hour_model.results.optimal_duals.get(f'balance constraint at hour {t}', 0)
    storage_profit += (discharge - charge) * mcp
print(f'Total Profit of Storage Unit: €{storage_profit:.2f}') 

total_generator_profit = {g: 0 for g in range(N_GENERATORS)}
for g in range(N_GENERATORS):
    for t in range(time_step):
        mcp = multi_hour_model.results.optimal_duals.get(f'balance constraint at hour {t}', 0)
        production = multi_hour_model.results.variables.get(f'production of generator {g} at hour {t}', 0)
        total_generator_profit[g] += (mcp - generators_combined['cost'].iloc[g]) * production

for g in range(N_GENERATORS):
    print(f'Profit of Generator {g+1} over 24 hours: €{total_generator_profit[g]:.2f}')

print(f"\n{'Hour':<6} {'MCP':>10} {'ServedDemand':>14} {'Generation':>12} {'Wind':>10} {'Charge':>10} {'Discharge':>10} {'SOC':>10} {'Balance':>10}")
for t in range(time_step):
    mcp = multi_hour_model.results.optimal_duals.get(f'balance constraint at hour {t}', 0)
    served_demand = sum(multi_hour_model.results.variables.get(f'demand of load {j} at hour {t}', 0) for j in range(N_LOADS))
    gen = sum(multi_hour_model.results.variables.get(f'production of generator {g} at hour {t}', 0) for g in GENERATORS)
    wind = sum(multi_hour_model.results.variables.get(f'production of generator {g} at hour {t}', 0) for g in WINDTURBINES)
    charge = multi_hour_model.results.variables.get(f'battery charge at hour {t}', 0)
    discharge = multi_hour_model.results.variables.get(f'battery discharge at hour {t}', 0)
    soc = multi_hour_model.results.variables.get(f'battery SOC at hour {t}', 0)
    balance_check = gen + discharge - charge - served_demand
    print(f"{t+1:<6} {mcp:>10.2f} {served_demand:>14.2f} {gen:>12.2f} {wind:>10.2f} {charge:>10.2f} {discharge:>10.2f} {soc:>10.2f} {balance_check:>10.4f}")

# Plots
# Plot 1: Market Clearing Price
mcp_by_hour = {t: multi_hour_model.results.optimal_duals.get(f'balance constraint at hour {t}', 0) for t in range(time_step)}

plt.figure(figsize=(12, 6))
plt.plot(list(mcp_by_hour.keys()), list(mcp_by_hour.values()), marker='o', linewidth=2, markersize=8, color='blue')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Market Clearing Price (€/MWh)', fontsize=12)
plt.title('Market Clearing Price Across 24 Hours', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(time_step))
plt.tight_layout()
plt.show()

# Plot 2: Generation and demand
served_demand_plot = []
conventional_generation = []
wind_generation = []
battery_soc_plot = []

for t in range(time_step):
    charge = multi_hour_model.results.variables.get(f'battery charge at hour {t}', 0)
    discharge = multi_hour_model.results.variables.get(f'battery discharge at hour {t}', 0)
    served_demand_plot.append(sum(multi_hour_model.results.variables.get(f'demand of load {j} at hour {t}', 0) for j in range(N_LOADS)))
    conventional_generation.append(sum(multi_hour_model.results.variables.get(f'production of generator {g} at hour {t}', 0) for g in GENERATORS))
    wind_generation.append(sum(multi_hour_model.results.variables.get(f'production of generator {g} at hour {t}', 0) for g in WINDTURBINES))
    battery_soc_plot.append(multi_hour_model.results.variables.get(f'battery SOC at hour {t}', 0))

hours_plot = list(range(time_step))

plt.figure(figsize=(12, 6))
plt.plot(hours_plot, load_capacity.values, marker='s', linewidth=2, color='green', label='Base Load')
plt.plot(hours_plot, served_demand_plot, marker='^', linewidth=2, color='orange', label='Served Demand')
plt.plot(hours_plot, conventional_generation, marker='o', linewidth=2, color='red', label='Total Generation')
plt.plot(hours_plot, wind_generation, marker='d', linewidth=2, color='blue', label='Wind Generation')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Power (MW)', fontsize=12)
plt.title('System Generation and Demand Across 24 Hours', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(hours_plot)
plt.tight_layout()
plt.show()

# Plot 3: Battery SOC
plt.figure(figsize=(12, 6))
plt.plot(hours_plot, battery_soc_plot, marker='o', linewidth=2, color='purple', label='Battery SOC')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('State of Charge (MWh)', fontsize=12)
plt.title('Battery State of Charge Across 24 Hours', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(hours_plot)
plt.tight_layout()
plt.show()