# step 6: Reserve Market + Day-Ahead Market (sequential clearing, European style)
# Builds upon Step 1: copper-plate, no storage, single hour

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

plots_dir = Path(__file__).parent / 'step6 plots'
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
# DATA LOADING
# ===========================================================================

date = '2019-08-31'

conventional_generators = pd.read_csv(
    'GeneratorsData.csv', header=None,
    names=['id', 'bus', 'capacity', 'cost', 'r_plus', 'r_minus']
)

assert not conventional_generators.isnull().any().any(), \
    f"NaN in GeneratorsData.csv:\n{conventional_generators[conventional_generators.isnull().any(axis=1)]}"

# Wind capacity matrix: 6 farms × 24 hours
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
    'r_plus':   0.0,   # wind farms do not participate in reserve
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
VOLL = 500  # €/MWh

# Hour selected (0-based)
HOUR = 8

# BSP list from external CSV
bsp_ids = pd.read_csv('bsp_generators.csv')['generator_id'].tolist()


# ===========================================================================
# BUILD FULL GENERATOR DATAFRAME FOR SELECTED HOUR
# (single concat — GENERATORS defined here, not repeated)
# ===========================================================================

wind_generator['capacity'] = wind_capacity[:, HOUR]
total_generators = pd.concat([conventional_generators, wind_generator], ignore_index=True)

assert total_generators.index.is_unique, "Duplicate indices in total_generators"

GENERATORS = range(len(total_generators))
LOADS      = range(len(load_distribution))

total_demand = loads['demand'][HOUR]

bsp_mask    = total_generators['id'].isin(bsp_ids)
bsp_indices = total_generators.index[bsp_mask].tolist()
bsp_data    = total_generators[bsp_mask].copy().reset_index(drop=True)

# Reserve bid prices
# Upward:   10% of marginal cost — opportunity cost of holding headroom unavailable for DA
# Downward:  5% of marginal cost — lower opportunity cost; unit stays online and keeps earning DA revenue
bsp_data['reserve_up_price']   = 0.10 * bsp_data['cost']
bsp_data['reserve_down_price'] = 0.05 * bsp_data['cost']

BSP = range(len(bsp_data))

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
        bid_prices.append(VOLL)

demand_data = pd.DataFrame({
    'node':             load_nodes,
    'bid_quantity_min': bid_quantities_min,
    'bid_quantity_max': bid_quantities_max,
    'bid_price':        bid_prices
})


# ===========================================================================
# STAGE 1 — RESERVE MARKET CLEARING
# Objective: minimize total reserve procurement cost
# Variables: r_up_g, r_dn_g  for each BSP g
# Constraints:
#   - system upward requirement  >= 15% * D_tot
#   - system downward requirement >= 10% * D_tot
#   - individual upward limit    r_up_g <= R_g^+
#   - individual downward limit  r_dn_g <= R_g^-
#   - joint capacity             r_up_g + r_dn_g <= P_max_g
# ===========================================================================

print('=' * 55)
print(f'STAGE 1 — Reserve Market Clearing  |  Hour {HOUR + 1}')
print('=' * 55)
print(f'  Total demand:        {total_demand:.2f} MW')
print(f'  Upward required:     {upward_reserve_req:.2f} MW  (15%)')
print(f'  Downward required:   {downward_reserve_req:.2f} MW  (10%)')
print(f'  BSP set:             {bsp_data["id"].tolist()}')
print(f'  Max upward capacity: {bsp_data["r_plus"].sum():.2f} MW')
print('-' * 55)

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
    model_name      = "Reserve Market Clearing"
)

model_reserve = LP_OptimizationProblem(input_data_reserve)
model_reserve.run()

if not hasattr(model_reserve.results, 'optimal_duals'):
    print(f"INFEASIBLE: max upward = {bsp_data['r_plus'].sum():.1f} MW, "
          f"required = {upward_reserve_req:.1f} MW")
    raise SystemExit("Reserve market infeasible. Add BSPs or change HOUR.")

reserve_price_up   = model_reserve.results.optimal_duals['upward reserve requirement']
reserve_price_down = model_reserve.results.optimal_duals['downward reserve requirement']

bsp_data['r_up_cleared']   = [model_reserve.results.variables[f'reserve up {g}']   for g in BSP]
bsp_data['r_down_cleared'] = [model_reserve.results.variables[f'reserve down {g}'] for g in BSP]

bsp_r_up   = dict(zip(bsp_data['id'], bsp_data['r_up_cleared']))
bsp_r_down = dict(zip(bsp_data['id'], bsp_data['r_down_cleared']))

print(f'  Reserve price (upward):   {reserve_price_up:.4f} €/MW')
print(f'  Reserve price (downward): {reserve_price_down:.4f} €/MW')
print(f'  Upward cleared:   {bsp_data["r_up_cleared"].sum():.2f} MW'
      f'  (required: {upward_reserve_req:.2f} MW)')
print(f'  Downward cleared: {bsp_data["r_down_cleared"].sum():.2f} MW'
      f'  (required: {downward_reserve_req:.2f} MW)')


# ===========================================================================
# STAGE 2 — DAY-AHEAD MARKET CLEARING (with reserve constraints)
# Identical to Step 1 except for BSP generators:
#   upper bound: p_g <= P_max_g - r_up*_g
#   lower bound: p_g >= r_down*_g
# Note: P_min not modeled, consistently with Step 1.
# ===========================================================================

print('=' * 55)
print(f'STAGE 2 — Day-Ahead Market Clearing  |  Hour {HOUR + 1}')
print('=' * 55)

def da_capacity_ub(g):
    r_up = bsp_r_up.get(total_generators['id'][g], 0.0)
    return total_generators['capacity'][g] - r_up

def da_capacity_lb(g):
    return bsp_r_down.get(total_generators['id'][g], 0.0)

input_data_da = LP_InputData(
    VARIABLES = [f'production of generator {g}' for g in GENERATORS] +
                [f'demand of load {j}'           for j in LOADS],

    CONSTRAINTS = ['balance constraint'] +
                  [f'capacity constraint {g}'    for g in GENERATORS] +
                  [f'reserve commitment min {g}' for g in bsp_indices] +
                  [f'demand min limit {j}'        for j in LOADS] +
                  [f'demand max limit {j}'        for j in LOADS],

    objective_coeff = {
        **{f'demand of load {j}':           demand_data['bid_price'][j]  for j in LOADS},
        **{f'production of generator {g}': -total_generators['cost'][g]  for g in GENERATORS},
    },

    constraints_coeff = {
        'balance constraint': {
            **{f'demand of load {j}':           1  for j in LOADS},
            **{f'production of generator {g}': -1  for g in GENERATORS},
        },
        **{f'capacity constraint {g}':
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
        **{f'capacity constraint {g}':    da_capacity_ub(g) for g in GENERATORS},
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

    objective_sense = GRB.MAXIMIZE,
    model_name      = "Day-Ahead Market Clearing (with Reserve)"
)

model_da = LP_OptimizationProblem(input_data_da)
model_da.run()

mcp_with_reserve = model_da.results.optimal_duals['balance constraint']


# ===========================================================================
# STEP 1 REFERENCE — DA WITHOUT RESERVE (for comparison)
# ===========================================================================

input_data_da_ref = LP_InputData(
    VARIABLES = [f'production of generator {g}' for g in GENERATORS] +
                [f'demand of load {j}'           for j in LOADS],

    CONSTRAINTS = ['balance constraint'] +
                  [f'capacity constraint {g}' for g in GENERATORS] +
                  [f'demand min limit {j}'    for j in LOADS] +
                  [f'demand max limit {j}'    for j in LOADS],

    objective_coeff = {
        **{f'demand of load {j}':           demand_data['bid_price'][j]  for j in LOADS},
        **{f'production of generator {g}': -total_generators['cost'][g]  for g in GENERATORS},
    },

    constraints_coeff = {
        'balance constraint': {
            **{f'demand of load {j}':           1  for j in LOADS},
            **{f'production of generator {g}': -1  for g in GENERATORS},
        },
        **{f'capacity constraint {g}':
               {f'production of generator {k}': int(k == g) for k in GENERATORS}
           for g in GENERATORS},
        **{f'demand min limit {j}': {f'demand of load {j}': 1} for j in LOADS},
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

    objective_sense = GRB.MAXIMIZE,
    model_name      = "Day-Ahead Market Clearing (no reserve, Step 1 reference)"
)

model_da_ref = LP_OptimizationProblem(input_data_da_ref)
model_da_ref.run()

mcp_no_reserve = model_da_ref.results.optimal_duals['balance constraint']


# ===========================================================================
# RESULTS — PRICES, SOCIAL WELFARE, OPERATING COST
# ===========================================================================

# Social welfare = objective value (sum of utilities minus generation costs)
sw_with_reserve = model_da.results.objective_value
sw_no_reserve   = model_da_ref.results.objective_value

# Operating cost = sum C_g * p_g (generation cost only)
op_cost_with_reserve = sum(
    total_generators['cost'][g] * model_da.results.variables[f'production of generator {g}']
    for g in GENERATORS
)
op_cost_no_reserve = sum(
    total_generators['cost'][g] * model_da_ref.results.variables[f'production of generator {g}']
    for g in GENERATORS
)

print(f'  DA MCP (no reserve):      €{mcp_no_reserve:.4f}/MWh')
print(f'  DA MCP (with reserve):    €{mcp_with_reserve:.4f}/MWh')
print(f'  MCP delta:                {mcp_with_reserve - mcp_no_reserve:+.4f} €/MWh')
print(f'  Reserve price (upward):   €{reserve_price_up:.4f}/MW')
print(f'  Reserve price (downward): €{reserve_price_down:.4f}/MW')
print('-' * 55)
print(f'  Social welfare (no reserve):   €{sw_no_reserve:.2f}')
print(f'  Social welfare (with reserve): €{sw_with_reserve:.2f}')
print(f'  SW loss due to reserve:        €{sw_no_reserve - sw_with_reserve:.2f}')
print(f'  Operating cost (no reserve):   €{op_cost_no_reserve:.2f}')
print(f'  Operating cost (with reserve): €{op_cost_with_reserve:.2f}')
print(f'  Op. cost increase:             €{op_cost_with_reserve - op_cost_no_reserve:+.2f}')


# ===========================================================================
# RESULTS — GENERATOR TABLE
# ===========================================================================

results_rows = []
for g in GENERATORS:
    gen_id  = total_generators['id'][g]
    is_bsp  = gen_id in bsp_ids
    p_ref   = model_da_ref.results.variables[f'production of generator {g}']
    p_rsv   = model_da.results.variables[f'production of generator {g}']
    r_up    = bsp_r_up.get(gen_id, 0.0)
    r_down  = bsp_r_down.get(gen_id, 0.0)

    profit_ref  = (mcp_no_reserve   - total_generators['cost'][g]) * p_ref
    profit_rsv  = (mcp_with_reserve - total_generators['cost'][g]) * p_rsv
    reserve_rev = reserve_price_up * r_up + reserve_price_down * r_down

    results_rows.append({
        'id':                gen_id,
        'is_bsp':            is_bsp,
        'capacity_MW':       total_generators['capacity'][g],
        'r_up_MW':           round(r_up,        4),
        'r_down_MW':         round(r_down,      4),
        'dispatch_no_rsv':   round(p_ref,       4),
        'dispatch_with_rsv': round(p_rsv,       4),
        'profit_no_rsv':     round(profit_ref,  4),
        'profit_with_rsv':   round(profit_rsv + reserve_rev, 4),
        'reserve_revenue':   round(reserve_rev, 4),
    })

results_df = pd.DataFrame(results_rows)

print('-' * 55)
print(f"{'Generator':<10} {'BSP':<6} {'r_up':>8} {'r_down':>8} "
      f"{'disp_ref':>10} {'disp_rsv':>10} "
      f"{'profit_ref':>12} {'profit_rsv':>12} {'rsv_rev':>10}")
print('-' * 95)
for _, row in results_df.iterrows():
    print(f"{str(row['id']):<10} {str(row['is_bsp']):<6} "
          f"{row['r_up_MW']:>8.2f} {row['r_down_MW']:>8.2f} "
          f"{row['dispatch_no_rsv']:>10.2f} {row['dispatch_with_rsv']:>10.2f} "
          f"{row['profit_no_rsv']:>12.2f} {row['profit_with_rsv']:>12.2f} "
          f"{row['reserve_revenue']:>10.2f}")

print('-' * 95)
print(f"{'TOTAL':<10} {'':<6} {results_df['r_up_MW'].sum():>8.2f} "
      f"{results_df['r_down_MW'].sum():>8.2f} "
      f"{results_df['dispatch_no_rsv'].sum():>10.2f} "
      f"{results_df['dispatch_with_rsv'].sum():>10.2f} "
      f"{results_df['profit_no_rsv'].sum():>12.2f} "
      f"{results_df['profit_with_rsv'].sum():>12.2f} "
      f"{results_df['reserve_revenue'].sum():>10.2f}")


# ===========================================================================
# PLOTS
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: dispatch comparison
x     = np.arange(len(results_df))
width = 0.35
axes[0].bar(x - width/2, results_df['dispatch_no_rsv'],   width,
            label='No reserve (Step 1)',   color='steelblue',  alpha=0.8)
axes[0].bar(x + width/2, results_df['dispatch_with_rsv'], width,
            label='With reserve (Step 6)', color='darkorange', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['id'], rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('Dispatch (MW)', fontweight='bold')
axes[0].set_title('Generator Dispatch: Step 1 vs Step 6', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Panel 2: reserve allocation per BSP
bsp_results = results_df[results_df['is_bsp']].copy()
x2 = np.arange(len(bsp_results))
axes[1].bar(x2 - width/2, bsp_results['r_up_MW'],   width,
            label='Reserve up (MW)',   color='green',   alpha=0.8)
axes[1].bar(x2 + width/2, bsp_results['r_down_MW'], width,
            label='Reserve down (MW)', color='crimson', alpha=0.8)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(bsp_results['id'], rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Reserve Capacity (MW)', fontweight='bold')
axes[1].set_title(
    f'Reserve Allocation per BSP\n'
    f'MCP: €{mcp_no_reserve:.2f} → €{mcp_with_reserve:.2f}/MWh',
    fontweight='bold'
)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / f'step6_results_hour_{HOUR + 1}.png', dpi=150, bbox_inches='tight')
plt.close()


# ===========================================================================
# CSV EXPORTS
# ===========================================================================

results_df.to_csv(plots_dir / 'step6_generator_results.csv', index=False)

pd.DataFrame([{
    'hour':                  HOUR + 1,
    'total_demand_MW':       total_demand,
    'upward_req_MW':         upward_reserve_req,
    'downward_req_MW':       downward_reserve_req,
    'reserve_price_up':      reserve_price_up,
    'reserve_price_down':    reserve_price_down,
    'mcp_no_reserve':        mcp_no_reserve,
    'mcp_with_reserve':      mcp_with_reserve,
    'mcp_delta':             mcp_with_reserve - mcp_no_reserve,
    'sw_no_reserve':         sw_no_reserve,
    'sw_with_reserve':       sw_with_reserve,
    'sw_loss':               sw_no_reserve - sw_with_reserve,
    'op_cost_no_reserve':    op_cost_no_reserve,
    'op_cost_with_reserve':  op_cost_with_reserve,
    'op_cost_increase':      op_cost_with_reserve - op_cost_no_reserve,
}]).to_csv(plots_dir / 'step6_prices.csv', index=False)

print(f"\nCSVs and plots saved to '{plots_dir.name}/'")