# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together!

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


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


# Define ranges and indexes
N_GENERATORS = 12 #number of generators
N_LOADS = 1 #number of inflexible loads
time_step = 24 #time step in hours (Delta_t)
GENERATORS = range(12) #range of generators
LOADS = range(1) #range of inflexible Loads

# Battery parameters
BATTERY_ENERGY_MAX = 100.0  # MWh
BATTERY_POWER_MAX = 50.0    # MW
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

def build_multi_hour_input_data(
    generator_cost,
    generator_capacity,
    load_capacity,
    generators_range,
    time_range,
):
    variables = [f'production of generator {g} at hour {t}' for t in time_range for g in generators_range]
    variables += [f'battery charge at hour {t}' for t in time_range]
    variables += [f'battery discharge at hour {t}' for t in time_range]
    variables += [f'battery soc at hour {t}' for t in time_range]
    constraints = [f'balance constraint at hour {t}' for t in time_range]
    constraints += [f'capacity constraint {g} at hour {t}' for t in time_range for g in generators_range]
    constraints += [f'battery soc balance at hour {t}' for t in time_range]
    constraints += [f'battery soc max at hour {t}' for t in time_range]
    constraints += [f'battery soc min at hour {t}' for t in time_range]
    constraints += [f'battery charge max at hour {t}' for t in time_range]
    constraints += [f'battery discharge max at hour {t}' for t in time_range]

    objective_coeff = {
        f'production of generator {g} at hour {t}': generator_cost[g]
        for t in time_range
        for g in generators_range
    }
    for t in time_range:
        objective_coeff[f'battery charge at hour {t}'] = 0.0
        objective_coeff[f'battery discharge at hour {t}'] = 0.0
        objective_coeff[f'battery soc at hour {t}'] = 0.0

    constraints_coeff = {}
    constraints_rhs = {}
    constraints_sense = {}

    for t in time_range:
        balance_name = f'balance constraint at hour {t}'
        constraints_coeff[balance_name] = {
            f'production of generator {g} at hour {t}': 1 for g in generators_range
        }
        constraints_coeff[balance_name][f'battery discharge at hour {t}'] = 1
        constraints_coeff[balance_name][f'battery charge at hour {t}'] = -1
        constraints_rhs[balance_name] = load_capacity[t]
        constraints_sense[balance_name] = GRB.EQUAL

        for g in generators_range:
            cap_name = f'capacity constraint {g} at hour {t}'
            constraints_coeff[cap_name] = {
                f'production of generator {g} at hour {t}': 1
            }
            constraints_rhs[cap_name] = generator_capacity[g]
            constraints_sense[cap_name] = GRB.LESS_EQUAL

        soc_balance_name = f'battery soc balance at hour {t}'
        constraints_coeff[soc_balance_name] = {
            f'battery soc at hour {t}': 1,
            f'battery charge at hour {t}': -BATTERY_ETA_CHARGE,
            f'battery discharge at hour {t}': 1 / BATTERY_ETA_DISCHARGE,
        }
        if t == 0:
            constraints_rhs[soc_balance_name] = BATTERY_INITIAL_SOC
        else:
            constraints_coeff[soc_balance_name][f'battery soc at hour {t - 1}'] = -1
            constraints_rhs[soc_balance_name] = 0.0
        constraints_sense[soc_balance_name] = GRB.EQUAL

        soc_max_name = f'battery soc max at hour {t}'
        constraints_coeff[soc_max_name] = {
            f'battery soc at hour {t}': 1
        }
        constraints_rhs[soc_max_name] = BATTERY_ENERGY_MAX
        constraints_sense[soc_max_name] = GRB.LESS_EQUAL

        soc_min_name = f'battery soc min at hour {t}'
        constraints_coeff[soc_min_name] = {
            f'battery soc at hour {t}': 1
        }
        constraints_rhs[soc_min_name] = 0.0
        constraints_sense[soc_min_name] = GRB.GREATER_EQUAL

        charge_max_name = f'battery charge max at hour {t}'
        constraints_coeff[charge_max_name] = {
            f'battery charge at hour {t}': 1
        }
        constraints_rhs[charge_max_name] = BATTERY_POWER_MAX
        constraints_sense[charge_max_name] = GRB.LESS_EQUAL

        discharge_max_name = f'battery discharge max at hour {t}'
        constraints_coeff[discharge_max_name] = {
            f'battery discharge at hour {t}': 1
        }
        constraints_rhs[discharge_max_name] = BATTERY_POWER_MAX
        constraints_sense[discharge_max_name] = GRB.LESS_EQUAL

    return LP_InputData(
        VARIABLES=variables,
        CONSTRAINTS=constraints,
        objective_coeff=objective_coeff,
        constraints_coeff=constraints_coeff,
        constraints_rhs=constraints_rhs,
        constraints_sense=constraints_sense,
        objective_sense=GRB.MINIMIZE,
        model_name="ED multi-hour problem",
    )

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


# Battery considerations: 
# The battery (Storage system) is located at the generator 10, since it has a cost of 0. So it can charge the battery when the prices are low.
# It is assumed the battery is of a capacity of 100 MWh. It is assumed it charges with a efficiency of 93 % and discharges with an efficiency of 95 %. The battery can charge and discharge at a maximum rate of 50 MW. 
# The constraints should be: 