

# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together!hwfehuafsh

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


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
                        gp.quicksum(self.data.constraints_coeff[c][v] * self.variables[v] for v in self.data.VARIABLES),
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
                print(f'Optimal value of {key}:', value)
        for key, value in self.results.optimal_duals.items():
                print(f'Dual variable of {key}:', value)

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
time_step = 1 #time step in hours (Delta_t)
GENERATORS = range(12) #range of generators
LOADS = range(1) #range of inflexible Loads

generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id','bus','capacity','cost'])

loads =pd.read_csv('LoadData.csv', header = None, names=['hour','demand'])

# Set values of input parameters
generator_cost = generators['cost'] # Variable generators costs (c_i)
generator_capacity = generators['capacity'] # Generators capacity (\Overline{P}_i)
generator_nodes = generators['bus'] # Nodes where generators are located (n_i)
#load_capacity =  loads['demand'] # Inflexible load demand (D_j)
load_capacity = 1775.835 # Inflexible load demand (D_j) for hour 1, as an example


VARIABLES = [f'production of generator {g}' for g in GENERATORS] # name of decision variables
CONSTRAINTS = ['balance constraint'] + [f'capacity constraint {g}' for g in GENERATORS] # name of constraints


objective_coeff = {VARIABLES[g]: generator_cost[g] for g in GENERATORS}  # Coefficients in objective function
constraints_coeff = {
    'balance constraint': {VARIABLES[g]: 1 for g in GENERATORS},
    **{f'capacity constraint {g}': {VARIABLES[k]: int(k == g) for k in GENERATORS} for g in GENERATORS}
}

# Right hand side coefficients of constraints
constraints_rhs = {
    'balance constraint': load_capacity,
    **{f'capacity constraint {g}': generator_capacity[g] for g in GENERATORS}
}

# Direction of constraints
constraints_sense = {
    'balance constraint': GRB.EQUAL,
    **{f'capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS}
}

objective_sense = GRB.MINIMIZE  # Optimization direction

model_name = "model1"  # name of model

model = LP_builder(
    VARIABLES,
    CONSTRAINTS,
    objective_coeff,
    constraints_coeff,
    constraints_rhs,
    constraints_sense,
    objective_sense,
    model_name
)

model.optimize()


