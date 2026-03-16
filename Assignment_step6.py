# step 6: Reserve Market + Day-Ahead Market (sequential clearing, European style)
# Builds upon Step 1: copper-plate, no storage, single hour

# imports
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# import os library to consider the same folder of this file for the csv reading
os.chdir(Path(__file__).parent)

# create output folder for plots
plots_dir = Path(__file__).parent / 'step6 plots'
plots_dir.mkdir(exist_ok=True)

# mute the gurobi license print
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()


class Expando(object):
    pass

# define classes for input data and optimization problem
class LP_InputData:

    def __init__(
        self,
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],              # Coefficients in objective function
        constraints_coeff: dict[str, dict[str, float]], # Linear coefficients of constraints
        constraints_rhs: dict[str, float],              # Right hand side coefficients of constraints
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
        self.results.variables = {v.VarName: v.x for v in self.model.getVars()}
        self.results.optimal_duals = {f'{c.ConstrName}': c.Pi for c in self.model.getConstrs()}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {self.model.ModelName} was not successful")


# Import data from case study
date = '2019-08-31' # Choose data for wind turbine generation

conventional_generators = pd.read_csv('GeneratorsData.csv', header=None,
                                      names=['id','bus','capacity','cost','r_plus','r_minus'])

# Creating the wind capacity matrix for 6 wind generators and 24 hours
wind_capacity = np.zeros((6, 24)) # placeholder for wind generator data, to be filled with actual data from CSV files for hourly optimization
file_list = sorted(Path(__file__).parent.glob('Ninja/*.csv'))
for i, csv in enumerate(file_list):
    data = pd.read_csv(csv, header=None, names=['time','local_time','capacity_factor'], skiprows=4)
    index = data.loc[data['time'] == date + ' 00:00'].index[0] # Find the index of the row corresponding to the specified date starting at 00:00
    wind_capacity[i, :] = data['capacity_factor'][index:index+24].values * 200

wind_generator = pd.DataFrame({ # wind generators data, with capacity to be updated for each hour based on CSV files
        'id': [f'wind_{i}' for i in range(wind_capacity.shape[0])],
        'bus': pd.read_csv('wind_farms.csv', usecols=['node'])['node'].values,
        'capacity': 0.0, # placeholder, will be updated for each hour
        'cost': [0.0 for i in range(wind_capacity.shape[0])]
    })

# Creating a single DataFrame with all generators (conventional + wind)
total_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)

# Load data upload
loads             = pd.read_csv('LoadData.csv', header=None, usecols=[1], names=['demand'])  # load data (hourly system demand)
load_distribution = pd.read_csv('load_distribution_1.csv')  # nodal load shares
load_nodes        = load_distribution['node'].tolist()  # list of all load nodes
load_percentages  = dict(zip(load_distribution['node'], load_distribution['pct_of_system_load'] / 100))  # fraction of total demand per node

# define the path and clear eventual spaces in the csv
df = pd.read_csv('elastic_data.csv')
df.columns = df.columns.str.strip()

# List of elastic loads: nodes 1, 7, 9, 13, 14, 15
elastic_nodes = df['node'].tolist()

# Bid prices for elastic loads (€/MWh) — differentiated, consistent with generation costs (€5.47–26.11/MWh)
elastic_bid_prices = df.set_index('node')['bid'].to_dict()
VOLL = 500  # Value of Lost Load (€/MWh), consistent across all steps

# Hour selected for reserve + DA market clearing (0-based index)
HOUR = 9

# BSP generator list (eligible conventional units)
bsp_ids = pd.read_csv('bsp_generators.csv')['generator_id'].tolist()

# Define ranges and indexes
GENERATORS = range(len(total_generators))
LOADS      = range(len(load_distribution))

# Update wind capacities and rebuild full generator DataFrame for selected hour
wind_generator['capacity'] = wind_capacity[:, HOUR]
total_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)

total_demand = loads['demand'][HOUR] # total demand for the selected hour

# Index positions of BSP generators within total_generators (needed for DA constraint building)
bsp_mask    = total_generators['id'].isin(bsp_ids)
bsp_indices = total_generators.index[bsp_mask].tolist()

# BSP sub-DataFrame (indexed by BSP = range(6))
bsp_data = total_generators[bsp_mask].copy().reset_index(drop=True)

# Reserve bid prices (€/MW of capacity held available, NOT €/MWh of energy delivered)
# Upward reserve:   10% of marginal cost (opportunity cost of keeping headroom)
# Downward reserve:  5% of marginal cost (lower opportunity cost — unit stays online)
bsp_data['reserve_up_price']   = 0.10 * bsp_data['cost']  # €/MW
bsp_data['reserve_down_price'] = 0.05 * bsp_data['cost']  # €/MW

BSP = range(len(bsp_data))

# Reserve requirements
# Upward:   15% of total demand
# Downward: 10% of total demand
upward_reserve_req   = 0.15 * total_demand
downward_reserve_req = 0.10 * total_demand

# Demand bid data (identical to Step 1)
bid_quantities_min, bid_quantities_max, bid_prices = [], [], []
for node in load_nodes:
    demand_at_node = total_demand * load_percentages[node]
    if node in elastic_nodes:
        bid_quantities_min.append(0)
        bid_quantities_max.append(demand_at_node)
        bid_prices.append(elastic_bid_prices[node])
    else:
        bid_quantities_min.append(demand_at_node)
        bid_quantities_max.append(demand_at_node)
        bid_prices.append(VOLL) # high bid price to ensure the inelastic load are always accepted (Value of Lost Load)

demand_data = pd.DataFrame({
    'node':             load_nodes,
    'bid_quantity_min': bid_quantities_min,
    'bid_quantity_max': bid_quantities_max,
    'bid_price':        bid_prices
})


# STAGE 1 — RESERVE MARKET CLEARING

print('-------------------')
print(f'Reserve Market Clearing — Hour {HOUR + 1}')
print('-------------------')

input_data = {
    'model_reserve': LP_InputData(
        VARIABLES = [f'reserve up {g}'   for g in BSP] +
                    [f'reserve down {g}' for g in BSP],

        CONSTRAINTS = ['upward reserve requirement',
                       'downward reserve requirement'] +
                      [f'reserve up capacity {g}'    for g in BSP] +
                      [f'reserve down capacity {g}'  for g in BSP] +
                      [f'reserve joint capacity {g}' for g in BSP],

        # Objective: minimize total reserve procurement cost (€/MW × MW committed = €).
        objective_coeff = {
            **{f'reserve up {g}':   bsp_data['reserve_up_price'][g]   for g in BSP},
            **{f'reserve down {g}': bsp_data['reserve_down_price'][g] for g in BSP},
        },

        constraints_coeff = {
            # System-level reserve requirements
            'upward reserve requirement':   {f'reserve up {g}':   1 for g in BSP},
            'downward reserve requirement': {f'reserve down {g}': 1 for g in BSP},
            # Individual BSP capacity limits (up and down separately)
            **{f'reserve up capacity {g}':   {f'reserve up {k}':   int(k == g) for k in BSP} for g in BSP},
            **{f'reserve down capacity {g}': {f'reserve down {k}': int(k == g) for k in BSP} for g in BSP},
            # Joint capacity: r_up_g + r_down_g <= P_max_g
            # A generator cannot commit more total reserve than its nameplate capacity
            **{f'reserve joint capacity {g}': {
                f'reserve up {g}':   1,
                f'reserve down {g}': 1
            } for g in BSP},
        },

       constraints_rhs = {
            'upward reserve requirement':   upward_reserve_req,
            'downward reserve requirement': downward_reserve_req,
            **{f'reserve up capacity {g}':    bsp_data['r_plus'][g]   for g in BSP},  # ← R_i^+
            **{f'reserve down capacity {g}':  bsp_data['r_minus'][g]  for g in BSP},  # ← R_i^-
            **{f'reserve joint capacity {g}': bsp_data['capacity'][g] for g in BSP},  # P_max invariato
        },

        constraints_sense = {
            'upward reserve requirement':   GRB.GREATER_EQUAL,
            'downward reserve requirement': GRB.GREATER_EQUAL,
            **{f'reserve up capacity {g}':    GRB.LESS_EQUAL for g in BSP},
            **{f'reserve down capacity {g}':  GRB.LESS_EQUAL for g in BSP},
            **{f'reserve joint capacity {g}': GRB.LESS_EQUAL for g in BSP},
        },

        objective_sense = GRB.MINIMIZE,
        model_name      = "Reserve Market Clearing"
    )
}

model_reserve = LP_OptimizationProblem(input_data['model_reserve'])
model_reserve.run()

# Reserve prices = dual variables of system-level requirement constraints
reserve_price_up   = model_reserve.results.optimal_duals['upward reserve requirement']
reserve_price_down = model_reserve.results.optimal_duals['downward reserve requirement']

# Cleared reserve quantities — fixed parameters for DA market
bsp_data['r_up_cleared']   = [model_reserve.results.variables[f'reserve up {g}']   for g in BSP]
bsp_data['r_down_cleared'] = [model_reserve.results.variables[f'reserve down {g}'] for g in BSP]

# Lookup dicts by generator id (used in DA constraint building)
bsp_r_up   = dict(zip(bsp_data['id'], bsp_data['r_up_cleared']))
bsp_r_down = dict(zip(bsp_data['id'], bsp_data['r_down_cleared']))

print(f'Reserve price (upward):   {reserve_price_up:.4f} €/MW')
print(f'Reserve price (downward): {reserve_price_down:.4f} €/MW')
print(f'Total upward reserve cleared:   {bsp_data["r_up_cleared"].sum():.2f} MW (required: {upward_reserve_req:.2f} MW)')
print(f'Total downward reserve cleared: {bsp_data["r_down_cleared"].sum():.2f} MW (required: {downward_reserve_req:.2f} MW)')
print('-------------------')


# ============================================================================
# STAGE 2 — DAY-AHEAD MARKET CLEARING (with reserve constraints)
# ============================================================================
# Identical to Step 1 except for BSP generators:
#   - Upper bound reduced: p_g <= P_max_g - r_up*_g  (reserved headroom unavailable for DA)
#   - Lower bound enforced: p_g >= r_down*_g           (must stay online to cover downward reserve)
# Non-BSP generators: unchanged (p_g <= P_max_g as in Step 1)

print(f'Day-Ahead Market Clearing — Hour {HOUR + 1} (post-reserve)')
print('-------------------')

def da_capacity_ub(g):
    """Effective DA upper bound for generator g after reserve commitment."""
    r_up = bsp_r_up.get(total_generators['id'][g], 0.0)
    return total_generators['capacity'][g] - r_up

def da_capacity_lb(g):
    """Effective DA lower bound for generator g (= r_down* for BSPs, 0 otherwise)."""
    return bsp_r_down.get(total_generators['id'][g], 0.0)

input_data = {
    'model_da': LP_InputData(
        VARIABLES = [f'production of generator {g}' for g in GENERATORS] +
                    [f'demand of load {j}'           for j in LOADS],

        CONSTRAINTS = ['balance constraint'] +
                      [f'capacity constraint {g}'        for g in GENERATORS] +
                      [f'reserve commitment min {g}'      for g in bsp_indices] +
                      [f'demand min limit {j}'            for j in LOADS] +
                      [f'demand max limit {j}'            for j in LOADS],

        objective_coeff = {
            # Demand utility (positive)
            **{f'demand of load {j}':            demand_data['bid_price'][j]   for j in LOADS},
            # Generation cost (negative)
            **{f'production of generator {g}':  -total_generators['cost'][g]  for g in GENERATORS},
        },

        constraints_coeff = {
            # Balance constraint: total generation must equal total demand
            'balance constraint': {
                **{f'demand of load {j}':           1  for j in LOADS},
                **{f'production of generator {g}': -1  for g in GENERATORS},
            },
            # Generator capacity (upper bound, reduced for BSPs)
            **{f'capacity constraint {g}':
                   {f'production of generator {k}': int(k == g) for k in GENERATORS}
               for g in GENERATORS},
            # Generator lower bound for BSPs: p_g >= r_down*
            **{f'reserve commitment min {g}':
                   {f'production of generator {k}': int(k == g) for k in GENERATORS}
               for g in bsp_indices},
            # Demand minimum limits
            **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
            # Demand maximum limits
            **{f'demand max limit {j}': {f'demand of load {j}': 1} for j in LOADS},
        },

        constraints_rhs = {
            'balance constraint': 0,
            # Upper bound: P_max - r_up* for BSPs, P_max for non-BSPs
            **{f'capacity constraint {g}':    da_capacity_ub(g) for g in GENERATORS},
            # Lower bound: r_down* for BSP generators only
            **{f'reserve commitment min {g}': da_capacity_lb(g) for g in bsp_indices},
            **{f'demand min limit {j}': demand_data['bid_quantity_min'][j] for j in LOADS},
            **{f'demand max limit {j}': demand_data['bid_quantity_max'][j] for j in LOADS},
        },

        constraints_sense = {
            'balance constraint':                           GRB.EQUAL,
            **{f'capacity constraint {g}':    GRB.LESS_EQUAL    for g in GENERATORS},
            **{f'reserve commitment min {g}': GRB.GREATER_EQUAL for g in bsp_indices},
            **{f'demand min limit {j}':       GRB.GREATER_EQUAL for j in LOADS},
            **{f'demand max limit {j}':       GRB.LESS_EQUAL    for j in LOADS},
        },

        objective_sense = GRB.MAXIMIZE, # maximize social welfare
        model_name      = "Day-Ahead Market Clearing (with Reserve)"
    )
}

model_da = LP_OptimizationProblem(input_data['model_da'])
model_da.run()

mcp_with_reserve = model_da.results.optimal_duals['balance constraint']
print(f'Day-Ahead MCP (with reserve):    {mcp_with_reserve:.4f} €/MWh')

# STEP 1 REFERENCE — DA CLEARING WITHOUT RESERVE (for comparison)
# Used to answer: "how does the reserve market change prices in the DA market?"

input_data = {
    'model_da_ref': LP_InputData(
        VARIABLES = [f'production of generator {g}' for g in GENERATORS] +
                    [f'demand of load {j}'           for j in LOADS],

        CONSTRAINTS = ['balance constraint'] +
                      [f'capacity constraint {g}' for g in GENERATORS] +
                      [f'demand min limit {j}'    for j in LOADS] +
                      [f'demand max limit {j}'    for j in LOADS],

        objective_coeff = {
            # Demand utility (positive)
            **{f'demand of load {j}':           demand_data['bid_price'][j]   for j in LOADS},
            # Generation cost (negative)
            **{f'production of generator {g}': -total_generators['cost'][g]   for g in GENERATORS},
        },

        constraints_coeff = {
            # Balance constraint: total generation must equal total demand
            'balance constraint': {
                **{f'demand of load {j}':           1  for j in LOADS},
                **{f'production of generator {g}': -1  for g in GENERATORS},
            },
            # Generator capacity
            **{f'capacity constraint {g}':
                   {f'production of generator {k}': int(k == g) for k in GENERATORS}
               for g in GENERATORS},
            # Demand minimum limits
            **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
            # Demand maximum limits
            **{f'demand max limit {j}': {f'demand of load {j}': 1} for j in LOADS},
        },

        constraints_rhs = {
            'balance constraint': 0,
            **{f'capacity constraint {g}': total_generators['capacity'][g]    for g in GENERATORS},
            **{f'demand min limit {j}':    demand_data['bid_quantity_min'][j] for j in LOADS},
            **{f'demand max limit {j}':    demand_data['bid_quantity_max'][j] for j in LOADS},
        },

        constraints_sense = {
            'balance constraint':                  GRB.EQUAL,
            **{f'capacity constraint {g}': GRB.LESS_EQUAL    for g in GENERATORS},
            **{f'demand min limit {j}':    GRB.GREATER_EQUAL for j in LOADS},
            **{f'demand max limit {j}':    GRB.LESS_EQUAL    for j in LOADS},
        },

        objective_sense = GRB.MAXIMIZE, # maximize social welfare
        model_name      = "Day-Ahead Market Clearing (no reserve, Step 1 reference)"
    )
}

model_da_ref = LP_OptimizationProblem(input_data['model_da_ref'])
model_da_ref.run()

mcp_no_reserve = model_da_ref.results.optimal_duals['balance constraint']
print(f'Day-Ahead MCP (no reserve, ref): {mcp_no_reserve:.4f} €/MWh')
print(f'MCP change due to reserve:       {mcp_with_reserve - mcp_no_reserve:+.4f} €/MWh')
print('-------------------')

# Results
results_rows = []
for g in GENERATORS:
    gen_id  = total_generators['id'][g]
    is_bsp  = gen_id in bsp_ids
    p_ref   = model_da_ref.results.variables[f'production of generator {g}']
    p_rsv   = model_da.results.variables[f'production of generator {g}']
    r_up    = bsp_r_up.get(gen_id, 0.0)
    r_down  = bsp_r_down.get(gen_id, 0.0)

    profit_ref  = mcp_no_reserve   * p_ref - total_generators['cost'][g] * p_ref
    profit_rsv  = mcp_with_reserve * p_rsv - total_generators['cost'][g] * p_rsv
    reserve_rev = reserve_price_up * r_up + reserve_price_down * r_down

    results_rows.append({
        'id':                gen_id,
        'is_bsp':            is_bsp,
        'capacity_MW':       total_generators['capacity'][g],
        'r_up_MW':           r_up,
        'r_down_MW':         r_down,
        'dispatch_no_rsv':   round(p_ref,  4),
        'dispatch_with_rsv': round(p_rsv,  4),
        'profit_no_rsv':     round(profit_ref,  4),
        'profit_with_rsv':   round(profit_rsv + reserve_rev, 4),
        'reserve_revenue':   round(reserve_rev, 4),
    })

results_df = pd.DataFrame(results_rows)

print(f'\nStep 6 market-clearing outcomes for hour {HOUR + 1}:')
print(f'Reserve price (upward):   €{reserve_price_up:.4f}/MW')
print(f'Reserve price (downward): €{reserve_price_down:.4f}/MW')
print(f'MCP without reserve: €{mcp_no_reserve:.2f}/MWh')
print(f'MCP with reserve:    €{mcp_with_reserve:.2f}/MWh')
print('\n')
print(f"{'Generator':<10} {'BSP':<6} {'r_up':>8} {'r_down':>8} {'disp_ref':>10} {'disp_rsv':>10} {'profit_ref':>12} {'profit_rsv':>12}")
for _, row in results_df.iterrows():
    print(f"{str(row['id']):<10} {str(row['is_bsp']):<6} {row['r_up_MW']:>8.2f} {row['r_down_MW']:>8.2f} "
          f"{row['dispatch_no_rsv']:>10.2f} {row['dispatch_with_rsv']:>10.2f} "
          f"{row['profit_no_rsv']:>12.2f} {row['profit_with_rsv']:>12.2f}")

# PLOTS

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: dispatch comparison with vs without reserve
x = np.arange(len(results_df))
width = 0.35
axes[0].bar(x - width/2, results_df['dispatch_no_rsv'],   width, label='No reserve (Step 1)', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, results_df['dispatch_with_rsv'], width, label='With reserve (Step 6)', color='darkorange', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['id'], rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('Dispatch (MW)', fontweight='bold')
axes[0].set_title('Generator Dispatch: Step 1 vs Step 6', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Panel 2: reserve allocation per BSP
bsp_results = results_df[results_df['is_bsp']].copy()
x2 = np.arange(len(bsp_results))
axes[1].bar(x2 - width/2, bsp_results['r_up_MW'],   width, label='Reserve up (MW)',   color='green',   alpha=0.8)
axes[1].bar(x2 + width/2, bsp_results['r_down_MW'], width, label='Reserve down (MW)', color='crimson', alpha=0.8)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(bsp_results['id'], rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Reserve Capacity (MW)', fontweight='bold')
axes[1].set_title(f'Reserve Allocation per BSP\nMCP: €{mcp_no_reserve:.2f} → €{mcp_with_reserve:.2f}/MWh', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / f'step6_results_hour_{HOUR + 1}.png', dpi=150, bbox_inches='tight')
plt.close()


# CSV Exports 

results_df.to_csv(plots_dir / 'step6_generator_results.csv', index=False)

print(f"CSVs saved to '{plots_dir.name}/'")
