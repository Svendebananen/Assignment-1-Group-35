"""
data_loader.py — Central data loading module for Assignment 1
=============================================================
Usage in any step script:
    from data_loader import load_all_data
    data = load_all_data()

    # Access data like:
    data['generators']          -> DataFrame with all generator info
    data['load_profile']        -> DataFrame with hourly system demand
    data['load_distribution']   -> DataFrame with nodal load shares
    data['lines']               -> DataFrame with transmission line data
    data['wind_farms']          -> DataFrame with wind farm locations & capacity
    data['wind_profiles']       -> DataFrame with hourly capacity factors per farm
    data['demand_bids']         -> DataFrame with hourly demand bid prices
    data['storage']             -> dict with storage parameters
    data['line_overrides']      -> DataFrame with modified line capacities (wind case)

You can also load individual pieces:
    from data_loader import load_generators, load_demand_for_hour
"""

import pandas as pd
import os

# ---------- Path setup ----------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------- Individual loaders ----------

def load_generators():
    """Load generator data (Tables 1 & 2 merged)."""
    return pd.read_csv(os.path.join(DATA_DIR, "generators.csv"))


def load_load_profile():
    """Load 24-hour system demand profile (Table 3)."""
    return pd.read_csv(os.path.join(DATA_DIR, "load_profile.csv"))


def load_load_distribution():
    """Load nodal demand shares (Table 4)."""
    return pd.read_csv(os.path.join(DATA_DIR, "load_distribution.csv"))


def load_lines(apply_wind_overrides=False):
    """Load transmission line data (Table 5).
    
    If apply_wind_overrides=True, reduces capacity on 3 lines
    as recommended in Section 3 of the IEEE paper for wind integration.
    """
    lines = pd.read_csv(os.path.join(DATA_DIR, "transmission_lines.csv"))
    if apply_wind_overrides:
        overrides = pd.read_csv(os.path.join(DATA_DIR, "line_capacity_overrides.csv"))
        for _, row in overrides.iterrows():
            mask = (
                (lines['from_node'] == row['from_node']) &
                (lines['to_node'] == row['to_node'])
            )
            lines.loc[mask, 'capacity_MVA'] = row['new_capacity_MVA']
    return lines


def load_wind_farms():
    """Load wind farm locations and capacities."""
    return pd.read_csv(os.path.join(DATA_DIR, "wind_farms.csv"))


def load_wind_profiles():
    """Load hourly wind capacity factors for each farm."""
    return pd.read_csv(os.path.join(DATA_DIR, "wind_profiles.csv"))


def load_demand_bids():
    """Load hourly demand bid prices."""
    return pd.read_csv(os.path.join(DATA_DIR, "demand_bid_prices.csv"))


def load_storage():
    """Load storage parameters as a dictionary."""
    df = pd.read_csv(os.path.join(DATA_DIR, "storage.csv"))
    params = {}
    for _, row in df.iterrows():
        val = row['value']
        params[row['parameter']] = float(val)
    return params


# ---------- Convenience functions ----------

def get_demand_at_hour(hour, load_profile=None, load_dist=None):
    """Get nodal demands (MW) for a specific hour.
    
    Returns a dict: {node: demand_MW}
    """
    if load_profile is None:
        load_profile = load_load_profile()
    if load_dist is None:
        load_dist = load_load_distribution()
    
    total = load_profile.loc[load_profile['hour'] == hour, 'system_demand_MW'].values[0]
    nodal = {}
    for _, row in load_dist.iterrows():
        nodal[int(row['node'])] = total * row['pct_of_system_load'] / 100.0
    return nodal


def get_wind_at_hour(hour, wind_farms=None, wind_profiles=None):
    """Get wind generation (MW) at each node for a specific hour.
    
    Returns a dict: {node: wind_MW}
    """
    if wind_farms is None:
        wind_farms = load_wind_farms()
    if wind_profiles is None:
        wind_profiles = load_wind_profiles()
    
    row = wind_profiles.loc[wind_profiles['hour'] == hour].iloc[0]
    result = {}
    for _, wf in wind_farms.iterrows():
        cf = row[f"wf{wf['wind_farm']}_cf"]
        result[int(wf['node'])] = wf['capacity_MW'] * cf
    return result


def get_total_wind_at_hour(hour):
    """Get total wind power available in the system for a given hour."""
    wind = get_wind_at_hour(hour)
    return sum(wind.values())


# ---------- Master loader ----------

def load_all_data(apply_wind_overrides=False):
    """Load everything into a single dict. One-stop shop."""
    return {
        'generators': load_generators(),
        'load_profile': load_load_profile(),
        'load_distribution': load_load_distribution(),
        'lines': load_lines(apply_wind_overrides=apply_wind_overrides),
        'wind_farms': load_wind_farms(),
        'wind_profiles': load_wind_profiles(),
        'demand_bids': load_demand_bids(),
        'storage': load_storage(),
        'line_overrides': pd.read_csv(os.path.join(DATA_DIR, "line_capacity_overrides.csv")),
    }


# ---------- Quick test ----------
if __name__ == "__main__":
    data = load_all_data()
    print("=== Data loaded successfully ===\n")
    
    print(f"Generators: {len(data['generators'])} units")
    print(f"  Conventional (cost > 0): {(data['generators']['Ci_dollar_MWh'] > 0).sum()}")
    print(f"  Renewable (cost = 0):    {(data['generators']['Ci_dollar_MWh'] == 0).sum()}")
    print(f"  Total capacity: {data['generators']['Pmax_MW'].sum()} MW\n")
    
    print(f"Wind farms: {len(data['wind_farms'])} farms")
    print(f"  Total wind capacity: {data['wind_farms']['capacity_MW'].sum()} MW\n")
    
    print(f"Demand nodes: {len(data['load_distribution'])}")
    print(f"  Peak demand: {data['load_profile']['system_demand_MW'].max()} MW")
    print(f"  Min demand:  {data['load_profile']['system_demand_MW'].min()} MW\n")
    
    print(f"Transmission lines: {len(data['lines'])}\n")
    
    print(f"Storage: Pch={data['storage']['Pch']} MW, E={data['storage']['E']} MWh\n")
    
    # Example: hour 10 data
    h = 10
    print(f"--- Hour {h} snapshot ---")
    print(f"  System demand: {data['load_profile'].loc[data['load_profile']['hour']==h, 'system_demand_MW'].values[0]:.1f} MW")
    print(f"  Total wind: {get_total_wind_at_hour(h):.1f} MW")
    nodal_demand = get_demand_at_hour(h)
    print(f"  Demand at node 15: {nodal_demand[15]:.1f} MW")
    print(f"  Demand bid price: {data['demand_bids'].loc[data['demand_bids']['hour']==h, 'bid_price_dollar_MWh'].values[0]} $/MWh")
