# step 6: Reserve Market + Day-Ahead Market joint clearing (US style)
# Builds upon Step 1: copper-plate, no storage, single hour
# step 6 extra: Joint Reserve + Day-Ahead Market (U.S.-style)
# Single LP co-optimizes energy and reserve simultaneously.
# Compare outcomes with European sequential clearing (Step 6).

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

plots_dir = Path(__file__).parent / 'step6 optional task plots'
plots_dir.mkdir(exist_ok=True)

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()


class Expando(object):
    pass


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
        self.VARIABLES         = VARIABLES
        self.CONSTRAINTS       = CONSTRAINTS
        self.objective_coeff   = objective_coeff
        self.constraints_coeff = constraints_coeff
        self.constraints_rhs   = constraints_rhs
        self.constraints_sense = constraints_sense
        self.objective_sense   = objective_sense
        self.model_name        = model_name


class LP_OptimizationProblem():
    def __init__(self, input_data: LP_InputData):
        self.data    = input_data
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        self.variables = {
            v: self.model.addVar(lb=0, name=f'{v}')
            for v in self.data.VARIABLES
        }

    def _build_constraints(self):
        self.constraints = {
            c: self.model.addLConstr(
                gp.quicksum(
                    self.data.constraints_coeff[c].get(v, 0) * self.variables[v]
                    for v in self.data.VARIABLES
                ),
                self.data.constraints_sense[c],
                self.data.constraints_rhs[c],
                name=f'{c}'
            )
            for c in self.data.CONSTRAINTS
        }

    def _build_objective_function(self):
        objective = gp.quicksum(
            self.data.objective_coeff[v] * self.variables[v]
            for v in self.data.VARIABLES
        )
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
        self.results.variables       = {v.VarName: v.x for v in self.model.getVars()}
        self.results.optimal_duals   = {c.ConstrName: c.Pi for c in self.model.getConstrs()}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {self.model.ModelName} was not successful")


# ===========================================================================
# DATA LOADING  (identical to step6.py)
# ===========================================================================

date = '2019-08-31'

conventional_generators = pd.read_csv(
    'GeneratorsData.csv', header=None,
    names=['id', 'bus', 'capacity', 'cost', 'r_plus', 'r_minus']
)

assert not conventional_generators.isnull().any().any(), \
    f"NaN in GeneratorsData.csv:\n{conventional_generators[conventional_generators.isnull().any(axis=1)]}"

wind_capacity = np.zeros((6, 24))
file_list = sorted(Path(__file__).parent.glob('Ninja/*.csv'))
for i, csv in enumerate(file_list):
    data  = pd.read_csv(csv, header=None,
                        names=['time', 'local_time', 'capacity_factor'], skiprows=4)
    index = data.loc[data['time'] == date + ' 00:00'].index[0]
    wind_capacity[i, :] = data['capacity_factor'][index:index + 24].values * 200

wind_generator = pd.DataFrame({
    'id':       [f'wind_{i}' for i in range(wind_capacity.shape[0])],
    'bus':      pd.read_csv('wind_farms.csv', usecols=['node'])['node'].values,
    'capacity': 0.0,
    'cost':     0.0,
    'r_plus':   0.0,
    'r_minus':  0.0,
})

loads             = pd.read_csv('LoadData.csv', header=None, usecols=[1], names=['demand'])
load_distribution = pd.read_csv('load_distribution_1.csv')
load_nodes        = load_distribution['node'].tolist()
load_percentages  = dict(zip(
    load_distribution['node'],
    load_distribution['pct_of_system_load'] / 100
))

df = pd.read_csv('elastic_data.csv')
df.columns = df.columns.str.strip()
elastic_nodes      = df['node'].tolist()
elastic_bid_prices = df.set_index('node')['bid'].to_dict()
VOLL = 500

HOUR = 8

bsp_ids = pd.read_csv('bsp_generators.csv')['generator_id'].tolist()

wind_generator['capacity'] = wind_capacity[:, HOUR]
total_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)

assert total_generators.index.is_unique, "Duplicate indices in total_generators"

GENERATORS = range(len(total_generators))
LOADS      = range(len(load_distribution))

total_demand = loads['demand'][HOUR]

bsp_mask    = total_generators['id'].isin(bsp_ids)
bsp_indices = total_generators.index[bsp_mask].tolist()
bsp_data    = total_generators[bsp_mask].copy().reset_index(drop=True)

bsp_data['reserve_up_price']   = 0.10 * bsp_data['cost']
bsp_data['reserve_down_price'] = 0.05 * bsp_data['cost']

BSP = range(len(bsp_data))

upward_reserve_req   = 0.15 * total_demand
downward_reserve_req = 0.10 * total_demand

# mapping: generator index g (in GENERATORS) → BSP index b (in BSP)
# used to cross-reference reserve variables in the capacity_ub constraint
gen_to_bsp = {g: b for b, g in enumerate(bsp_indices)}

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
        bid_prices.append(VOLL)

demand_data = pd.DataFrame({
    'node':             load_nodes,
    'bid_quantity_min': bid_quantities_min,
    'bid_quantity_max': bid_quantities_max,
    'bid_price':        bid_prices
})


# ===========================================================================
# EUROPEAN SEQUENTIAL CLEARING  (Step 6 reference — reproduced here for
# comparison without reading from CSV, consistent with Step 6 convention)
# ===========================================================================

# --- Stage 1: reserve market ---
input_data_reserve = LP_InputData(
    VARIABLES = [f'reserve up {g}'   for g in BSP] +
                [f'reserve down {g}' for g in BSP],

    CONSTRAINTS = ['upward reserve requirement',
                   'downward reserve requirement'] +
                  [f'reserve up capacity {g}'    for g in BSP] +
                  [f'reserve down capacity {g}'  for g in BSP] +
                  [f'reserve joint capacity {g}' for g in BSP],

    objective_coeff = {
        **{f'reserve up {g}':   bsp_data['reserve_up_price'][g]   for g in BSP},
        **{f'reserve down {g}': bsp_data['reserve_down_price'][g] for g in BSP},
    },

    constraints_coeff = {
        'upward reserve requirement':   {f'reserve up {g}':   1 for g in BSP},
        'downward reserve requirement': {f'reserve down {g}': 1 for g in BSP},
        **{f'reserve up capacity {g}':
               {f'reserve up {k}': int(k == g) for k in BSP}
           for g in BSP},
        **{f'reserve down capacity {g}':
               {f'reserve down {k}': int(k == g) for k in BSP}
           for g in BSP},
        **{f'reserve joint capacity {g}': {
               f'reserve up {g}':   1,
               f'reserve down {g}': 1,
           } for g in BSP},
    },

    constraints_rhs = {
        'upward reserve requirement':   upward_reserve_req,
        'downward reserve requirement': downward_reserve_req,
        **{f'reserve up capacity {g}':    bsp_data['r_plus'][g]   for g in BSP},
        **{f'reserve down capacity {g}':  bsp_data['r_minus'][g]  for g in BSP},
        **{f'reserve joint capacity {g}': bsp_data['capacity'][g] for g in BSP},
    },

    constraints_sense = {
        'upward reserve requirement':   GRB.EQUAL,
        'downward reserve requirement': GRB.EQUAL,
        **{f'reserve up capacity {g}':    GRB.LESS_EQUAL for g in BSP},
        **{f'reserve down capacity {g}':  GRB.LESS_EQUAL for g in BSP},
        **{f'reserve joint capacity {g}': GRB.LESS_EQUAL for g in BSP},
    },

    objective_sense = GRB.MINIMIZE,
    model_name      = "European Reserve Market"
)

model_reserve = LP_OptimizationProblem(input_data_reserve)
model_reserve.run()

eu_reserve_price_up   = model_reserve.results.optimal_duals['upward reserve requirement']
eu_reserve_price_down = model_reserve.results.optimal_duals['downward reserve requirement']

bsp_data['r_up_eu']   = [model_reserve.results.variables[f'reserve up {g}']   for g in BSP]
bsp_data['r_down_eu'] = [model_reserve.results.variables[f'reserve down {g}'] for g in BSP]

bsp_r_up_eu   = dict(zip(bsp_data['id'], bsp_data['r_up_eu']))
bsp_r_down_eu = dict(zip(bsp_data['id'], bsp_data['r_down_eu']))

# --- Stage 2: DA market with fixed reserve ---
def da_ub_eu(g):
    return total_generators['capacity'][g] - bsp_r_up_eu.get(total_generators['id'][g], 0.0)

def da_lb_eu(g):
    return bsp_r_down_eu.get(total_generators['id'][g], 0.0)

input_data_da_eu = LP_InputData(
    VARIABLES = [f'production of generator {g}' for g in GENERATORS] +
                [f'demand of load {j}'           for j in LOADS],

    CONSTRAINTS = ['balance constraint'] +
                  [f'capacity ub {g}'            for g in GENERATORS] +
                  [f'reserve commitment min {g}' for g in bsp_indices] +
                  [f'demand min limit {j}'        for j in LOADS] +
                  [f'demand max limit {j}'        for j in LOADS],

    objective_coeff = {
        **{f'demand of load {j}':           demand_data['bid_price'][j]  for j in LOADS},
        **{f'production of generator {g}': -total_generators['cost'][g]  for g in GENERATORS},
    },

    constraints_coeff = {
        'balance constraint': {
            **{f'demand of load {j}':            1 for j in LOADS},
            **{f'production of generator {g}':  -1 for g in GENERATORS},
        },
        **{f'capacity ub {g}':
               {f'production of generator {k}': int(k == g) for k in GENERATORS}
           for g in GENERATORS},
        **{f'reserve commitment min {g}':
               {f'production of generator {k}': int(k == g) for k in GENERATORS}
           for g in bsp_indices},
        **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
        **{f'demand max limit {j}': {f'demand of load {j}': 1} for j in LOADS},
    },

    constraints_rhs = {
        'balance constraint': 0,
        **{f'capacity ub {g}':            da_ub_eu(g)                           for g in GENERATORS},
        **{f'reserve commitment min {g}': da_lb_eu(g)                           for g in bsp_indices},
        **{f'demand min limit {j}':       demand_data['bid_quantity_min'][j]    for j in LOADS},
        **{f'demand max limit {j}':       demand_data['bid_quantity_max'][j]    for j in LOADS},
    },

    constraints_sense = {
        'balance constraint':                             GRB.EQUAL,
        **{f'capacity ub {g}':            GRB.LESS_EQUAL    for g in GENERATORS},
        **{f'reserve commitment min {g}': GRB.GREATER_EQUAL for g in bsp_indices},
        **{f'demand min limit {j}':       GRB.GREATER_EQUAL for j in LOADS},
        **{f'demand max limit {j}':       GRB.LESS_EQUAL    for j in LOADS},
    },

    objective_sense = GRB.MAXIMIZE,
    model_name      = "European DA Market"
)

model_da_eu = LP_OptimizationProblem(input_data_da_eu)
model_da_eu.run()

eu_mcp = model_da_eu.results.optimal_duals['balance constraint']


# ===========================================================================
# U.S.-STYLE JOINT CLEARING
# Single LP: maximize SW - reserve procurement cost
#
# Key structural differences vs. European model:
#   1. Reserve variables r_up_g, r_down_g are optimization variables, not
#      parameters fixed from a prior stage.
#   2. Objective subtracts reserve costs (- c_up * r_up - c_down * r_down).
#   3. The capacity upper bound is now a linear constraint in two variables:
#         p_g + r_up_g <= P_max_g    (for BSP generators)
#      instead of the parametric bound:
#         p_g <= P_max_g - r_up_g*   (European Stage 2)
#   4. The minimum dispatch constraint is also linear in two variables:
#         p_g - r_down_g >= 0        (for BSP generators)
#      instead of the parametric bound:
#         p_g >= r_down_g*           (European Stage 2)
#   5. Three prices are extracted from a single LP as dual variables.
# ===========================================================================

print('=' * 60)
print(f'U.S.-STYLE JOINT CLEARING  |  Hour {HOUR + 1}')
print('=' * 60)

# Build capacity_ub constraint coefficients.
# For non-BSP generators: only p_g appears   → {production_g: 1}
# For BSP generators:     p_g + r_up_b ≤ P_max → {production_g: 1, reserve_up_b: 1}
def cap_ub_coeff(g):
    coeff = {f'production of generator {g}': 1}
    if g in gen_to_bsp:
        coeff[f'reserve up {gen_to_bsp[g]}'] = 1
    return coeff

# Build reserve_commitment_min constraint coefficients.
# p_g - r_down_b >= 0   →   {production_g: 1, reserve_down_b: -1}
def res_min_coeff(g):
    b = gen_to_bsp[g]
    return {
        f'production of generator {g}':  1,
        f'reserve down {b}':            -1,
    }

input_data_joint = LP_InputData(
    VARIABLES = [f'production of generator {g}' for g in GENERATORS] +
                [f'demand of load {j}'           for j in LOADS]      +
                [f'reserve up {g}'               for g in BSP]        +
                [f'reserve down {g}'             for g in BSP],

    CONSTRAINTS = ['balance constraint',
                   'upward reserve requirement',
                   'downward reserve requirement']                     +
                  [f'capacity ub {g}'            for g in GENERATORS] +
                  [f'reserve commitment min {g}' for g in bsp_indices]+
                  [f'reserve up capacity {g}'    for g in BSP]        +
                  [f'reserve down capacity {g}'  for g in BSP]        +
                  [f'reserve joint capacity {g}' for g in BSP]        +
                  [f'demand min limit {j}'        for j in LOADS]     +
                  [f'demand max limit {j}'        for j in LOADS],

    # Objective: maximize SW_DA - reserve_procurement_cost
    # Reserve terms enter with negative sign inside a maximization,
    # which is equivalent to minimizing their cost.
    objective_coeff = {
        **{f'demand of load {j}':           demand_data['bid_price'][j]         for j in LOADS},
        **{f'production of generator {g}': -total_generators['cost'][g]         for g in GENERATORS},
        **{f'reserve up {g}':              -bsp_data['reserve_up_price'][g]     for g in BSP},
        **{f'reserve down {g}':            -bsp_data['reserve_down_price'][g]   for g in BSP},
    },

    constraints_coeff = {
        'balance constraint': {
            **{f'demand of load {j}':           1  for j in LOADS},
            **{f'production of generator {g}': -1  for g in GENERATORS},
        },
        # reserve requirements: reserve variables only
        'upward reserve requirement':   {f'reserve up {g}':   1 for g in BSP},
        'downward reserve requirement': {f'reserve down {g}': 1 for g in BSP},

        # capacity upper bound: p_g + r_up_b <= P_max_g  (BSPs)
        #                       p_g           <= P_max_g  (non-BSPs)
        **{f'capacity ub {g}': cap_ub_coeff(g) for g in GENERATORS},

        # minimum dispatch: p_g - r_down_b >= 0  (BSPs only)
        **{f'reserve commitment min {g}': res_min_coeff(g) for g in bsp_indices},

        # individual reserve capacity bounds
        **{f'reserve up capacity {g}':
               {f'reserve up {k}': int(k == g) for k in BSP}
           for g in BSP},
        **{f'reserve down capacity {g}':
               {f'reserve down {k}': int(k == g) for k in BSP}
           for g in BSP},

        # joint capacity constraint: r_up_g + r_down_g <= P_max_g
        **{f'reserve joint capacity {g}': {
               f'reserve up {g}':   1,
               f'reserve down {g}': 1,
           } for g in BSP},

        **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
        **{f'demand max limit {j}': {f'demand of load {j}': 1} for j in LOADS},
    },

    constraints_rhs = {
        'balance constraint':           0,
        'upward reserve requirement':   upward_reserve_req,
        'downward reserve requirement': downward_reserve_req,

        # RHS is now only P_max (r_up is a variable on the LHS)
        **{f'capacity ub {g}':            total_generators['capacity'][g]       for g in GENERATORS},
        # RHS is 0 (r_down is a variable on the LHS)
        **{f'reserve commitment min {g}': 0                                     for g in bsp_indices},

        **{f'reserve up capacity {g}':    bsp_data['r_plus'][g]                 for g in BSP},
        **{f'reserve down capacity {g}':  bsp_data['r_minus'][g]                for g in BSP},
        **{f'reserve joint capacity {g}': bsp_data['capacity'][g]               for g in BSP},

        **{f'demand min limit {j}': demand_data['bid_quantity_min'][j]          for j in LOADS},
        **{f'demand max limit {j}': demand_data['bid_quantity_max'][j]          for j in LOADS},
    },

    constraints_sense = {
        'balance constraint':                             GRB.EQUAL,
        'upward reserve requirement':                     GRB.EQUAL,
        'downward reserve requirement':                   GRB.EQUAL,
        **{f'capacity ub {g}':            GRB.LESS_EQUAL    for g in GENERATORS},
        **{f'reserve commitment min {g}': GRB.GREATER_EQUAL for g in bsp_indices},
        **{f'reserve up capacity {g}':    GRB.LESS_EQUAL    for g in BSP},
        **{f'reserve down capacity {g}':  GRB.LESS_EQUAL    for g in BSP},
        **{f'reserve joint capacity {g}': GRB.LESS_EQUAL    for g in BSP},
        **{f'demand min limit {j}':       GRB.GREATER_EQUAL for j in LOADS},
        **{f'demand max limit {j}':       GRB.LESS_EQUAL    for j in LOADS},
    },

    objective_sense = GRB.MAXIMIZE,
    model_name      = "Joint Reserve + DA Market (U.S.-style)"
)

model_joint = LP_OptimizationProblem(input_data_joint)
model_joint.run()

us_mcp            = model_joint.results.optimal_duals['balance constraint']
us_reserve_up     = model_joint.results.optimal_duals['upward reserve requirement']
us_reserve_down   = model_joint.results.optimal_duals['downward reserve requirement']

bsp_data['r_up_us']   = [model_joint.results.variables[f'reserve up {g}']   for g in BSP]
bsp_data['r_down_us'] = [model_joint.results.variables[f'reserve down {g}'] for g in BSP]


# ===========================================================================
# RESULTS
# ===========================================================================

# Social welfare
sw_eu = model_da_eu.results.objective_value
sw_us = model_joint.results.objective_value  # includes reserve cost

# Reserve procurement cost
rsv_cost_eu = sum(
    bsp_data['reserve_up_price'][g]   * bsp_data['r_up_eu'][g] +
    bsp_data['reserve_down_price'][g] * bsp_data['r_down_eu'][g]
    for g in BSP
)
rsv_cost_us = sum(
    bsp_data['reserve_up_price'][g]   * bsp_data['r_up_us'][g] +
    bsp_data['reserve_down_price'][g] * bsp_data['r_down_us'][g]
    for g in BSP
)

# Net social welfare (DA welfare minus reserve cost) — comparable across models
net_sw_eu = sw_eu - rsv_cost_eu
net_sw_us = sw_us   # already net in joint objective

print(f'\n  {"Metric":<35} {"European":>12} {"U.S. joint":>12}')
print(f'  {"-"*60}')
print(f'  {"DA MCP (€/MWh)":<35} {eu_mcp:>12.4f} {us_mcp:>12.4f}')
print(f'  {"Reserve price UP (€/MW)":<35} {eu_reserve_price_up:>12.4f} {us_reserve_up:>12.4f}')
print(f'  {"Reserve price DOWN (€/MW)":<35} {eu_reserve_price_down:>12.4f} {us_reserve_down:>12.4f}')
print(f'  {"Reserve cost (€)":<35} {rsv_cost_eu:>12.2f} {rsv_cost_us:>12.2f}')
print(f'  {"DA social welfare (€)":<35} {sw_eu:>12.2f} {sw_us:>12.2f}')
print(f'  {"Net social welfare (€)":<35} {net_sw_eu:>12.2f} {net_sw_us:>12.2f}')
print(f'  {"Net SW gain U.S. vs EU (€)":<35} {"":>12} {net_sw_us - net_sw_eu:>+12.2f}')

print(f'\n  {"Generator":<10} {"r_up EU":>9} {"r_up US":>9} {"r_dn EU":>9} {"r_dn US":>9}')
print(f'  {"-"*50}')
for g in BSP:
    print(f'  {bsp_data["id"][g]:<10} '
          f'{bsp_data["r_up_eu"][g]:>9.2f} {bsp_data["r_up_us"][g]:>9.2f} '
          f'{bsp_data["r_down_eu"][g]:>9.2f} {bsp_data["r_down_us"][g]:>9.2f}')


# ===========================================================================
# PLOTS
# ===========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Step 6 Extra — European vs. U.S.-style  |  Hour {HOUR + 1}',
             fontweight='bold', fontsize=13)

labels   = ['European\n(sequential)', 'U.S.\n(joint)']
colors   = ['steelblue', 'darkorange']

# Panel 1: DA MCP comparison
axes[0].bar(labels, [eu_mcp, us_mcp], color=colors, alpha=0.85, width=0.4)
axes[0].set_ylabel('DA MCP (€/MWh)', fontweight='bold')
axes[0].set_title('Day-Ahead Market Clearing Price', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([eu_mcp, us_mcp]):
    axes[0].text(i, v + 0.05, f'€{v:.2f}', ha='center', fontsize=10, fontweight='bold')

# Panel 2: reserve allocation comparison per BSP
x     = np.arange(len(bsp_data))
width = 0.2
axes[1].bar(x - 1.5*width, bsp_data['r_up_eu'],   width, label='r_up EU',   color='steelblue',  alpha=0.85)
axes[1].bar(x - 0.5*width, bsp_data['r_up_us'],   width, label='r_up US',   color='darkorange', alpha=0.85)
axes[1].bar(x + 0.5*width, bsp_data['r_down_eu'], width, label='r_dn EU',   color='steelblue',  alpha=0.45)
axes[1].bar(x + 1.5*width, bsp_data['r_down_us'], width, label='r_dn US',   color='darkorange', alpha=0.45)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bsp_data['id'], rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Reserve capacity (MW)', fontweight='bold')
axes[1].set_title('Reserve Allocation per BSP', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')

# Panel 3: net social welfare comparison
net_sw_values = [net_sw_eu, net_sw_us]
bars = axes[2].bar(labels, net_sw_values, color=colors, alpha=0.85, width=0.4)
axes[2].set_ylabel('Net Social Welfare (€)', fontweight='bold')
axes[2].set_title('Net Social Welfare\n(DA welfare − reserve cost)', fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(net_sw_values):
    axes[2].text(i, v + 50, f'€{v:,.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / f'step6_extra_EU_vs_US_hour_{HOUR + 1}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to '{plots_dir.name}/'")


# ===========================================================================
# CSV EXPORT
# ===========================================================================

pd.DataFrame([{
    'hour':                HOUR + 1,
    'eu_mcp':              eu_mcp,
    'us_mcp':              us_mcp,
    'eu_reserve_price_up': eu_reserve_price_up,
    'us_reserve_price_up': us_reserve_up,
    'eu_reserve_price_dn': eu_reserve_price_down,
    'us_reserve_price_dn': us_reserve_down,
    'eu_reserve_cost':     rsv_cost_eu,
    'us_reserve_cost':     rsv_cost_us,
    'eu_da_sw':            sw_eu,
    'us_da_sw':            sw_us,
    'eu_net_sw':           net_sw_eu,
    'us_net_sw':           net_sw_us,
    'net_sw_gain':         net_sw_us - net_sw_eu,
}]).to_csv(plots_dir / 'step6_extra_comparison.csv', index=False)

bsp_export = bsp_data[['id', 'capacity', 'r_plus', 'r_minus',
                        'r_up_eu', 'r_down_eu', 'r_up_us', 'r_down_us']].copy()
bsp_export.to_csv(plots_dir / 'step6_extra_bsp_reserves.csv', index=False)

print(f"CSVs saved to '{plots_dir.name}/'")