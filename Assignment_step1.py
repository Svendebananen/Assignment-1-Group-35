# Howdy partner
# ! Welcome to the wild west of coding. Let's wrangle some code together!

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import glob
import numpy as np
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
            label = key
            if key.startswith("production of generator "):
                suffix = key.split(" ")[-1]
                if suffix.isdigit():
                    label = f"production of generator {int(suffix) + 1}"
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
N_GENERATORS = 24 #number of generators (18 conventional + 6 wind)
N_LOADS = 1 #number of inflexible loads
time_step = 24 #time step in hours (Delta_t)
GENERATORS = range(24) #range of generators (18 conventional + 6 wind)
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
#load_capacity =  loads['demand'] # Inflexible load demand (D_j)
load_capacity = loads['demand'] # Inflexible load demand (D_j) for hour 1, as an example


# Merit order curve for hour 5 (index 4)
hour_idx = 4  # Hour 5 (0-indexed)
demand_hour5 = load_capacity[hour_idx]
wind_supply_hour5 = wind_generators[:, hour_idx]

# Collect all generators with their costs and capacities for hour 5
gen_data = []

# Conventional generators
for i in range(len(generators)):
    gen_data.append({
        'name': f"Conv Gen {i}",
        'cost': generators['cost'].iloc[i],
        'capacity': generators['capacity'].iloc[i],
        'type': 'conventional'
    })

# Wind generators (cost = 0)
for i in range(len(wind_supply_hour5)):
    gen_data.append({
        'name': f"Wind Farm {i}",
        'cost': 0,
        'capacity': wind_supply_hour5[i],
        'type': 'wind'
    })

# Sort by cost (merit order)
gen_data_sorted = sorted(gen_data, key=lambda x: x['cost'])

# Calculate cumulative capacity
cumulative_capacity = [0]
costs_for_plot = [0]
for gen in gen_data_sorted:
    cumulative_capacity.append(cumulative_capacity[-1] + gen['capacity'])
    costs_for_plot.append(gen['cost'])
    cumulative_capacity.append(cumulative_capacity[-1])
    costs_for_plot.append(gen['cost'])

# Remove the last duplicate point
cumulative_capacity.pop()
costs_for_plot.pop()

# Create the merit order curve plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the merit order curve
ax.step(cumulative_capacity, costs_for_plot, where='post', linewidth=2.5, color='steelblue', label='Merit Order Curve')
ax.fill_between(cumulative_capacity, costs_for_plot, step='post', alpha=0.3, color='steelblue')

# Add demand line
ax.axvline(x=demand_hour5, color='red', linestyle='--', linewidth=2.5, label=f'Demand: {demand_hour5:.2f} MW')

# Find and mark the equilibrium point
equilibrium_cost = None
for i, cum_cap in enumerate(cumulative_capacity):
    if cum_cap >= demand_hour5:
        equilibrium_cost = costs_for_plot[i]
        break

if equilibrium_cost is not None:
    ax.plot(demand_hour5, equilibrium_cost, 'ro', markersize=10, label=f'Equilibrium Price: €{equilibrium_cost:.2f}/MWh')
    ax.axhline(y=equilibrium_cost, color='red', linestyle=':', alpha=0.5)

# Formatting
ax.set_xlabel('Cumulative Capacity (MW)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cost (€/MWh)', fontsize=12, fontweight='bold')
ax.set_title(f'Merit Order Curve - Hour 5', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()


for t in range(time_step):  # Loop over time steps (hours)
    print(f'------------------- {t + 1}  -------------------')
    print(load_capacity[t])

    # Reset generators to original conventional generators for each time step
    generators = pd.read_csv('GeneratorsData.csv', header=None, names=['id','bus','capacity','cost'])
    
    wind_generators_at_t = wind_generators[:,t] # Extracting wind generator data for hour t
    
    wind_df = pd.DataFrame({
        'id': [f'wind_{i}' for i in range(6)],
        'bus': wind_bus,
        'capacity': wind_generators[:,t],
        'cost': 0.0
    })

    generators = pd.concat([generators, wind_df], ignore_index=True)
    generator_cost = generators['cost'] # Variable generators costs (c_i)
    generator_capacity = generators['capacity'] # Generators capacity (\Overline{P}_i)
    generator_nodes = generators['bus'] # Nodes where generators are located (n_i)
    #load_capacity =  loads['demand'] # Inflexible load demand (D_j)

    
    input_data = {
        'model0': LP_InputData(
            VARIABLES = [f'production of generator {g}' for g in GENERATORS], 
            CONSTRAINTS = ['balance constraint'] + [f'capacity constraint {g}' for g in GENERATORS], 
            objective_coeff = {f'production of generator {g}': generator_cost[g] for g in GENERATORS}, 
            constraints_coeff = {'balance constraint': {f'production of generator {g}': 1 for g in GENERATORS},**{f'capacity constraint {g}': {f'production of generator {k}': int(k == g) for k in GENERATORS} for g in GENERATORS}},
            constraints_rhs = {'balance constraint': load_capacity[t],**{f'capacity constraint {g}': generator_capacity[g] for g in GENERATORS}},
            constraints_sense = {'balance constraint': GRB.EQUAL,**{f'capacity constraint {g}': GRB.LESS_EQUAL for g in GENERATORS}},
            objective_sense = GRB.MINIMIZE,
            model_name = "ED problem"
     )
    }
    model = LP_OptimizationProblem(input_data['model0'])
    model.run()
    model.display_results()
    print(f'--------------------------------------------------')
