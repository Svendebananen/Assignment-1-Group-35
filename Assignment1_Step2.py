# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together!

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class LP_InputData:

    def __init__(
        self, 
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],               # Coefficients in objective function
        constraints_coeff: dict[str, dict[str,float]],    # Linear coefficients of constraints
        constraints_rhs: dict[str, float],                # Right hand side coefficients of constraints
        constraints_sense: dict[str, int],              # Direction of constraints
        objective_sense: int,                           # Direction of op2timization
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
            print(f"optimization of {model.ModelName} was not successful")
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Optimal objective:", self.results.objective_value)
        for key, value in self.results.variables.items():
            label = key
            if key.startswith("production of generator "):
                if " at hour " in key:
                    prefix, hour_part = key.split(" at hour ", 1)
                    gen_suffix = prefix.split(" ")[-1]
                    hour_suffix = hour_part.split(" ")[0]
                    if gen_suffix.isdigit() and hour_suffix.isdigit():
                        label = f"production of generator {int(gen_suffix) + 1} at hour {int(hour_suffix) + 1}"
                else:
                    suffix = key.split(" ")[-1]
                    if suffix.isdigit():
                        label = f"production of generator {int(suffix) + 1}"
            print(f'Optimal value of {label}:', value)
        for key, value in self.results.optimal_duals.items():
            label = key
            if key.startswith("capacity constraint "):
                if " at hour " in key:
                    prefix, hour_part = key.split(" at hour ", 1)
                    gen_suffix = prefix.split(" ")[-1]
                    hour_suffix = hour_part.split(" ")[0]
                    if gen_suffix.isdigit() and hour_suffix.isdigit():
                        label = f"capacity constraint {int(gen_suffix) + 1} at hour {int(hour_suffix) + 1}"
                else:
                    suffix = key.split(" ")[-1]
                    if suffix.isdigit():
                        label = f"capacity constraint {int(suffix) + 1}"
            print(f'Dual variable of {label}:', value)

def LP_builder(
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],               # Coefficients in objective function
        constraints_coeff: dict[str, dict[str,float]],    # Linear coefficients of constraints
        constraints_rhs: dict[str, float],                # Right hand side coefficients of constraints
        constraints_sense: dict[str, int],              # Direction of constraints
        objective_sense: int,                           # Direction of op2timization
        model_name: str                                 # Name of model
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


def build_multi_hour_input_data(
    generator_cost,
    generator_capacity,
    load_capacity,
    generators_range,
    time_range,
):
    # Generator variables: production at each hour
    variables = [f'production of generator {g} at hour {t}' for t in time_range for g in generators_range]
    
    # Battery variables: charge power, discharge power, and state of charge
    variables += [f'battery charge at hour {t}' for t in time_range]
    variables += [f'battery discharge at hour {t}' for t in time_range]
    variables += [f'battery SOC at hour {t}' for t in time_range]
    
    # Balance constraints
    constraints = [f'balance constraint at hour {t}' for t in time_range]
    
    # Generator capacity constraints
    constraints += [f'capacity constraint {g} at hour {t}' for t in time_range for g in generators_range]
    
    # Battery power limits
    constraints += [f'battery charge limit at hour {t}' for t in time_range]
    constraints += [f'battery discharge limit at hour {t}' for t in time_range]
    
    # Battery energy constraints
    constraints += [f'battery SOC limit at hour {t}' for t in time_range]
    
    # Battery state of charge dynamics
    constraints += [f'battery dynamics at hour {t}' for t in time_range]

    # Objective coefficients (only generators have cost, battery is free)
    objective_coeff = {
        f'production of generator {g} at hour {t}': generator_cost[g]
        for t in time_range
        for g in generators_range
    }
    
    # Battery variables have zero cost
    for t in time_range:
        objective_coeff[f'battery charge at hour {t}'] = 0
        objective_coeff[f'battery discharge at hour {t}'] = 0
        objective_coeff[f'battery SOC at hour {t}'] = 0

    constraints_coeff = {}
    constraints_rhs = {}
    constraints_sense = {}

    for t in time_range:
        # Balance constraint: generators + battery discharge = load + battery charge
        balance_name = f'balance constraint at hour {t}'
        constraints_coeff[balance_name] = {
            f'production of generator {g} at hour {t}': 1 for g in generators_range
        }
        constraints_coeff[balance_name][f'battery discharge at hour {t}'] = 1
        constraints_coeff[balance_name][f'battery charge at hour {t}'] = -1
        constraints_rhs[balance_name] = load_capacity[t]
        constraints_sense[balance_name] = GRB.EQUAL

        # Generator capacity constraints
        for g in generators_range:
            cap_name = f'capacity constraint {g} at hour {t}'
            constraints_coeff[cap_name] = {
                f'production of generator {g} at hour {t}': 1
            }
            constraints_rhs[cap_name] = generator_capacity[g]
            constraints_sense[cap_name] = GRB.LESS_EQUAL

        # Battery charge limit
        batt_charge_limit = f'battery charge limit at hour {t}'
        constraints_coeff[batt_charge_limit] = {
            f'battery charge at hour {t}': 1
        }
        constraints_rhs[batt_charge_limit] = BATTERY_POWER_MAX_CHARGE
        constraints_sense[batt_charge_limit] = GRB.LESS_EQUAL

        # Battery discharge limit
        batt_disch_limit = f'battery discharge limit at hour {t}'
        constraints_coeff[batt_disch_limit] = {
            f'battery discharge at hour {t}': 1
        }
        constraints_rhs[batt_disch_limit] = BATTERY_POWER_MAX_DISCHARGE
        constraints_sense[batt_disch_limit] = GRB.LESS_EQUAL

        # Battery SOC limit (state of charge <= capacity)
        batt_soc_limit = f'battery SOC limit at hour {t}'
        constraints_coeff[batt_soc_limit] = {
            f'battery SOC at hour {t}': 1
        }
        constraints_rhs[batt_soc_limit] = BATTERY_ENERGY_MAX
        constraints_sense[batt_soc_limit] = GRB.LESS_EQUAL

        # Battery dynamics: SOC_t = SOC_{t-1} + eta_ch * P_ch_t - (1/eta_disch) * P_disch_t
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
            # First hour uses initial SOC
            constraints_rhs[batt_dyn] = BATTERY_INITIAL_SOC

        constraints_sense[batt_dyn] = GRB.EQUAL

    return LP_InputData(
        VARIABLES=variables,
        CONSTRAINTS=constraints,
        objective_coeff=objective_coeff,
        constraints_coeff=constraints_coeff,
        constraints_rhs=constraints_rhs,
        constraints_sense=constraints_sense,
        objective_sense=GRB.MINIMIZE,
        model_name="ED multi-hour problem with battery",
    )

# Define ranges and indexes
N_GENERATORS = 12 #number of generators
N_LOADS = 1 #number of inflexible loads
time_step = 24 #time step in hours (Delta_t)
GENERATORS = range(12) #range of generators
LOADS = range(1) #range of inflexible Loads

# Battery parameters
BATTERY_ENERGY_MAX = 100.0  # MWh
BATTERY_POWER_MAX_DISCHARGE = 50.0    # MW
BATTERY_POWER_MAX_CHARGE = 50.0    # MW
BATTERY_ETA_CHARGE = 0.93
BATTERY_ETA_DISCHARGE = 0.95
BATTERY_INITIAL_SOC = 0.0   # MWh

generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id','bus','capacity','cost'])

loads =pd.read_csv('LoadData.csv', header = None, names=['hour','demand'])

# Set values of input parameters
generator_cost = generators['cost'] # Variable generators costs (c_i)
generator_capacity = generators['capacity'] # Generators capacity (\Overline{P}_i)
generator_nodes = generators['bus'] # Nodes where generators are located (n_i)
#load_capacity =  loads['demand'] # Inflexible load demand (D_j)
load_capacity = loads['demand'] # Inflexible load demand (D_j) for hour 1, as an example

# Multi-hour model sketch (run this instead of the loop above if you want a single model)
multi_hour_data = build_multi_hour_input_data(
    generator_cost=generator_cost,
    generator_capacity=generator_capacity,
    load_capacity=load_capacity,
    generators_range=GENERATORS,
    time_range=range(time_step),
)
multi_hour_model = LP_OptimizationProblem(multi_hour_data)
multi_hour_model.run()
multi_hour_model.display_results()

# Extract and plot market clearing price (MCP)
mcp_by_hour = {}
for t in range(time_step):
    balance_constraint_name = f'balance constraint at hour {t}'
    if balance_constraint_name in multi_hour_model.results.optimal_duals:
        mcp_by_hour[t] = multi_hour_model.results.optimal_duals[balance_constraint_name]

# Create plot
plt.figure(figsize=(12, 6))
hours = list(mcp_by_hour.keys())
prices = list(mcp_by_hour.values())

plt.plot(hours, prices, marker='o', linewidth=2, markersize=8, color='blue')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Market Clearing Price ($/MWh)', fontsize=12)
plt.title('Market Clearing Price Across 24 Hours', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(hours)
plt.tight_layout()
plt.show()

# Plot system demand (load) across 24 hours
plt.figure(figsize=(12, 6))
hours_load = list(range(time_step))
load_values = load_capacity.values

# Extract battery charge for each hour and add to total demand
total_demand = []
conventional_generation = []
for t in range(time_step):
    base_load = load_capacity[t]
    battery_charge_key = f'battery charge at hour {t}'
    battery_charge = multi_hour_model.results.variables.get(battery_charge_key, 0)
    total_demand.append(base_load + battery_charge)
    
    # Sum conventional generation from all generators
    gen_production = 0
    for g in GENERATORS:
        gen_key = f'production of generator {g} at hour {t}'
        gen_production += multi_hour_model.results.variables.get(gen_key, 0)
    conventional_generation.append(gen_production)

plt.plot(hours_load, load_values, marker='s', linewidth=2, markersize=8, color='green', label='Base Load')
plt.plot(hours_load, total_demand, marker='^', linewidth=2, markersize=8, color='orange', label='Total Demand (with Battery Charging)')
plt.plot(hours_load, conventional_generation, marker='o', linewidth=2, markersize=8, color='red', label='Conventional Generation')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('System Demand (MWh)', fontsize=12)
plt.title('System Demand and Conventional Generation Across 24 Hours', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(hours_load)
plt.tight_layout()
plt.show()

# Battery considerations: 
# The battery (Storage system) is located at the generator 10, since it has a cost of 0. So it can charge the battery when the prices are low.
# It is assumed the battery is of a capacity of 100 MWh. It is assumed it charges with a efficiency of 93 % and discharges with an efficiency of 95 %. The battery can charge and discharge at a maximum rate of 50 MW. 
# The constraints should be: 
