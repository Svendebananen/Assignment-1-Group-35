# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together!

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import glob
import numpy as np


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
    def __init__(self, input_data: LP_InputData):
        self.data = input_data
        self.results = Expando()
        self._build_model()
    
    def _build_variables(self):
        self.variables = {v: self.model.addVar(lb=0, name=v) for v in self.data.VARIABLES}
    
    def _build_constraints(self):
        for c in self.data.CONSTRAINTS:
            self.model.addLConstr(
                gp.quicksum(self.data.constraints_coeff[c][v] * self.variables[v] for v in self.data.VARIABLES),
                self.data.constraints_sense[c],
                self.data.constraints_rhs[c],
                name=c
            )

    def _build_objective_function(self):
        obj = gp.quicksum(self.data.objective_coeff[v] * self.variables[v] for v in self.data.VARIABLES)
        self.model.setObjective(obj, self.data.objective_sense)

    def _build_model(self):
        self.model = gp.Model(name=self.data.model_name)
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()
    
    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.variables = {v.VarName: v.x for v in self.model.getVars()}
        self.results.optimal_duals = {c.ConstrName: c.Pi for c in self.model.getConstrs()}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"Optimization failed for {self.model.ModelName}")

    def display_results(self):
        if self.model.status == GRB.OPTIMAL:
            print()
            print('-------------------   RESULTS   -------------------')
            print(f"Model Name: {self.data.model_name}")
            print("Optimal objective:", self.results.objective_value)
            print("\nDECISION VARIABLES:")
            for var_name, value in self.results.variables.items():
                print(f'Optimal value of {var_name}:', value)
            print("\nDUAL VARIABLES (Shadow Prices):")
            for constr_name, pi in self.results.optimal_duals.items():
                print(f'Dual variable of {constr_name}:', pi)
        else:
            print("Optimization was not successful, no results to display.")



# Define ranges and indexes
N_GENERATORS = 18 #number of generators
N_LOADS = 1 #number of inflexible loads
time_step = 24 #time step in hours (Delta_t)
GENERATORS = range(18) #range of generators 
LOADS = range(1) #range of inflexible Loads

generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id','bus','capacity','cost'])

wind_generators= np.zeros((6, 24)) # Placeholder for wind generator data, to be filled with actual data from CSV files
file_list = glob.glob(r'C:\Users\User\Assignment-1-Group-35\Ninja\*.csv')
for i,csv in enumerate(file_list):
  data = pd.read_csv(csv, header=None,names = ['time','local_time','capacity_factor'],skiprows =4)
  wind_generators[i,:] = data['capacity_factor'][5808:5808+24].values*200 # Extracting data for 24 hours (assuming data is hourly and starts at index 5813)

wind_bus = pd.read_csv('wind_farms.csv',usecols=['node'])['node'].values
loads =pd.read_csv('LoadData.csv', header = None, names=['hour','demand'])


# Set values of input parameters
generator_cost = generators['cost'] # Variable generators costs (c_i)
generator_capacity = generators['capacity'] # Generators capacity (\Overline{P}_i)
generator_nodes = generators['bus'] # Nodes where generators are located (n_i)
load_capacity = loads['demand'] # Inflexible load demand (D_j) for hour 1, as an example




for t in range(time_step):
    print(f'\n--- HOUR {t + 1} ---')
    
    
    wind_df = pd.DataFrame({
        'id': [f'wind_{i}' for i in range(6)],
        'capacity': wind_generators[:, t],
        'cost': 0.0
    })
    
    
    current_generators = pd.concat([generators, wind_df], ignore_index=True)
    
   
    TOTAL_GENS = range(len(current_generators))
    
    
    input_obj = LP_InputData(
        VARIABLES = [f'gen_{g}' for g in TOTAL_GENS],
        CONSTRAINTS = ['balance'] + [f'cap_{g}' for g in TOTAL_GENS],
        objective_coeff = {f'gen_{g}': current_generators['cost'][g] for g in TOTAL_GENS},
        constraints_coeff = {
            'balance': {f'gen_{g}': 1 for g in TOTAL_GENS},
            **{f'cap_{g}': {f'gen_{k}': int(k == g) for k in TOTAL_GENS} for g in TOTAL_GENS}
        },
        constraints_rhs = {
            'balance': load_capacity[t],
            **{f'cap_{g}': current_generators['capacity'][g] for g in TOTAL_GENS}
        },
        constraints_sense = {
            'balance': GRB.EQUAL,
            **{f'cap_{g}': GRB.LESS_EQUAL for g in TOTAL_GENS}
        },
        objective_sense = GRB.MINIMIZE,
        model_name = f"ED_Hour_{t}"
    )

   
    problem = LP_OptimizationProblem(input_obj)
    problem.run()
    problem.display_results()
print(f'--------------------------------------------------')
