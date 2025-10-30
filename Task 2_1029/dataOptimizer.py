import pandas as pd
from gurobipy import Model, GRB, quicksum

# ============================================
# Configuration and constants
# ============================================

FACILITY_TYPES = ["small", "medium", "large"]
SEGMENT_BOUNDS = [
    (0.0, 0.10),
    (0.10, 0.15),
    (0.15, 0.20)
]
COST_FACTORS = [200, 400, 1000]


# ============================================
# Utility functions
# ============================================

def get_capacity(facility_type):
    if facility_type == "small":
        return 100
    elif facility_type == "medium":
        return 200
    return 400


def get_05_capacity(facility_type):
    if facility_type == "small":
        return 50
    elif facility_type == "medium":
        return 100
    return 200


def get_build_cost(facility_type):
    if facility_type == "small":
        return 65000.0
    elif facility_type == "medium":
        return 95000.0
    return 115000.0


def get_cost_factor(segment_idx):
    return COST_FACTORS[segment_idx]


# ============================================
# Data loading
# ============================================

def load_data(data_file, distance_file, too_close_file, facility_data_file, facility_zip_map_file):
    """Load all data files and prepare basic structures."""
    data = pd.read_csv(data_file)
    distances = pd.read_csv(distance_file)
    too_close = pd.read_csv(too_close_file)
    facility_data = pd.read_csv(facility_data_file)
    facility_zip_map = pd.read_csv(facility_zip_map_file)

    data["zip_code"] = data["zip_code"].astype(int)
    facility_data["zip_code"] = facility_data["zip_code"].astype(int)
    facility_zip_map["zip_code"] = facility_zip_map["zip_code"].astype(int)

    if "zip_code" in distances.columns:
        distances["zip_code"] = distances["zip_code"].astype(int)
    if "zip_code" in too_close.columns:
        too_close["zip_code"] = too_close["zip_code"].astype(int)

    zips = sorted(data["zip_code"].unique())
    facilities = facility_data["facility_id"].tolist()

    potential_locations = {}
    for z in zips:
        row = data[data["zip_code"] == z].iloc[0]
        num_locs = int(row["potential_locations"]) if "potential_locations" in row.index else 0
        potential_locations[z] = list(range(num_locs))

    return data, distances, too_close, facility_data, facility_zip_map, zips, facilities, potential_locations


# ============================================
# Model building
# ============================================

def create_variables(model, facilities, facility_data, zips, potential_locations):
    """Create all Gurobi variables."""
    x, delta, w, y = {}, {}, {}, {}

    for facility_id in facilities:
        facility_row = facility_data[facility_data["facility_id"] == facility_id]
        if facility_row.empty:
            continue
        initial_capacity = float(facility_row["initial_capacity_0_12"].values[0])
        if initial_capacity > 0:
            x[facility_id] = model.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"expand_{facility_id}")
            for k in range(len(SEGMENT_BOUNDS)):
                delta[facility_id, k] = model.addVar(vtype=GRB.BINARY, name=f"delta_{facility_id}_{k}")
                w[facility_id, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{facility_id}_{k}")
        else:
            x[facility_id] = 0.0

    for z in zips:
        if z in potential_locations and potential_locations[z]:
            for l in potential_locations[z]:
                for t in FACILITY_TYPES:
                    y[z, l, t] = model.addVar(vtype=GRB.BINARY, name=f"build_{z}_{l}_{t}")

    return x, delta, w, y


def add_constraints(model, data, distances, too_close, facility_data, zips, facilities, potential_locations, x, delta, w, y):
    """Add all constraints."""
    for facility_id in facilities:
        facility_row = facility_data[facility_data["facility_id"] == facility_id]
        if facility_row.empty:
            continue
        initial_capacity = float(facility_row["initial_capacity_0_12"].values[0])
        if initial_capacity <= 0:
            continue

        model.addConstr(quicksum(delta[facility_id, k] for k in range(len(SEGMENT_BOUNDS))) == 1,
                        name=f"delta_sum_{facility_id}")

        M = max(initial_capacity * 0.2, 1.0)
        for k, (lower, upper) in enumerate(SEGMENT_BOUNDS):
            model.addConstr(x[facility_id] >= lower * initial_capacity * delta[facility_id, k])
            model.addConstr(x[facility_id] <= upper * initial_capacity * delta[facility_id, k] + M * (1 - delta[facility_id, k]))
            model.addConstr(w[facility_id, k] <= x[facility_id])
            model.addConstr(w[facility_id, k] <= M * delta[facility_id, k])
            model.addConstr(w[facility_id, k] >= x[facility_id] - M * (1 - delta[facility_id, k]))

    # Coverage constraints
    for z in zips:
        zip_row = data[data["zip_code"] == z].iloc[0]
        current_012 = float(zip_row.get("existing_capacity_0_12", 0.0))
        req_012 = float(zip_row.get("min_required_0_12", 0.0))
        facilities_in_zip = facility_data[facility_data["zip_code"] == z]["facility_id"].tolist()

        total_expansion = quicksum(x[f] for f in facilities_in_zip if f in x)
        new_capacity = quicksum(
            get_capacity(t) * y[z, l, t]
            for l in potential_locations[z]
            for t in FACILITY_TYPES
            if (z, l, t) in y
        ) if z in potential_locations else 0

        model.addConstr(current_012 + total_expansion + new_capacity >= req_012)

        current_05 = float(zip_row.get("existing_capacity_0_5", 0.0))
        req_05 = float(zip_row.get("min_required_0_5", 0.0))
        total_expansion_05 = total_expansion
        new_05 = quicksum(
            get_05_capacity(t) * y[z, l, t]
            for l in potential_locations[z]
            for t in FACILITY_TYPES
            if (z, l, t) in y
        ) if z in potential_locations else 0
        model.addConstr(current_05 + total_expansion_05 + new_05 >= req_05)

    # Distance constraints
    if distances is not None and not distances.empty:
        close_pairs = distances[distances["too_close"] == True] if "too_close" in distances.columns else distances
        for _, row in close_pairs.iterrows():
            z = int(row["zip_code"])
            loc1, loc2 = int(row["loc1_id"]), int(row["loc2_id"])
            if z not in potential_locations:
                continue
            if loc1 not in potential_locations[z] or loc2 not in potential_locations[z]:
                continue
            for t in FACILITY_TYPES:
                if (z, loc1, t) in y and (z, loc2, t) in y:
                    model.addConstr(y[z, loc1, t] + y[z, loc2, t] <= 1)

    if too_close is not None and not too_close.empty:
        for _, row in too_close.iterrows():
            z, loc = int(row["zip_code"]), int(row["location_id"])
            if z not in potential_locations or loc not in potential_locations[z]:
                continue
            for t in FACILITY_TYPES:
                if (z, loc, t) in y:
                    model.addConstr(y[z, loc, t] == 0)

    for z in zips:
        if z in potential_locations and potential_locations[z]:
            for l in potential_locations[z]:
                model.addConstr(
                    quicksum(y[z, l, t] for t in FACILITY_TYPES if (z, l, t) in y) <= 1
                )


def set_objective(model, facilities, facility_data, zips, potential_locations, y, w):
    """Set objective: minimize total cost."""
    build_cost = quicksum(
        get_build_cost(t) * y[z, l, t]
        for z in zips
        if z in potential_locations
        for l in potential_locations[z]
        for t in FACILITY_TYPES
        if (z, l, t) in y
    )

    equip_cost = quicksum(
        100.0 * get_05_capacity(t) * y[z, l, t]
        for z in zips
        if z in potential_locations
        for l in potential_locations[z]
        for t in FACILITY_TYPES
        if (z, l, t) in y
    )

    expand_cost_terms = []
    for facility_id in facilities:
        facility_row = facility_data[facility_data["facility_id"] == facility_id]
        if facility_row.empty:
            continue
        initial_capacity = float(facility_row["initial_capacity_0_12"].values[0])
        if initial_capacity <= 0:
            continue
        for k in range(len(SEGMENT_BOUNDS)):
            cost_factor = 20000.0 + get_cost_factor(k) * initial_capacity
            expand_cost_terms.append((cost_factor / initial_capacity) * w[facility_id, k])

    expand_cost = quicksum(expand_cost_terms) if expand_cost_terms else 0.0
    model.setObjective(build_cost + equip_cost + expand_cost, GRB.MINIMIZE)


# ============================================
# Solving
# ============================================

def solve_model(model, output_flag=1, mip_gap=0.05, time_limit=3600):
    """Solve the model and handle non-optimal cases."""
    model.setParam('OutputFlag', int(output_flag))
    model.setParam('MIPGap', float(mip_gap))
    model.setParam('TimeLimit', float(time_limit))
    model.optimize()
    return model


# ============================================
# Result extraction
# ============================================

def extract_results(model, data, facility_data, zips, facilities, potential_locations, x, delta, y):
    """Extract results into separate facility-level and ZIP-level DataFrames."""
    zip_results = []
    expanded_facilities = []
    new_facilities = []

    for z in zips:
        row = data[data["zip_code"] == z].iloc[0]
        facilities_in_zip = facility_data[facility_data["zip_code"] == z]
        total_expand = 0.0

        # ---------- Existing facilities (expansions) ----------
        for _, facility in facilities_in_zip.iterrows():
            fid = facility["facility_id"]
            x_val = float(x[fid].X) if hasattr(x[fid], "X") else float(x[fid])
            if x_val > 1e-6:  
                seg = "N/A"
                for k in range(len(SEGMENT_BOUNDS)):
                    if (fid, k) in delta and hasattr(delta[fid, k], "X") and delta[fid, k].X > 0.5:
                        seg = f"{int(SEGMENT_BOUNDS[k][0]*100)}%-{int(SEGMENT_BOUNDS[k][1]*100)}%"
                        break
                expanded_facilities.append({
                    "zip_code": z,
                    "facility_id": fid,
                    "expand_amount": x_val,
                    "expand_segment": seg,
                    "initial_capacity": float(facility["initial_capacity_0_12"]),
                    "final_capacity_est": float(facility["initial_capacity_0_12"]) + x_val
                })
            total_expand += x_val

        # ---------- New facilities ----------
        small = medium = large = 0
        if z in potential_locations and potential_locations[z]:
            for l in potential_locations[z]:
                for t in FACILITY_TYPES:
                    if (z, l, t) in y and hasattr(y[z, l, t], "X") and y[z, l, t].X > 0.5:
                        new_facilities.append({
                            "zip_code": z,
                            "location_id": l,
                            "facility_type": t,
                            "capacity_0_12": get_capacity(t),
                            "capacity_0_5": get_05_capacity(t),
                            "build_cost": get_build_cost(t),
                            "equipment_cost": 100.0 * get_05_capacity(t)
                        })
                        if t == "small": small += 1
                        elif t == "medium": medium += 1
                        else: large += 1

        total_new = small * get_capacity("small") + medium * get_capacity("medium") + large * get_capacity("large")

        # ---------- ZIP-level summary ----------
        zip_results.append({
            "zip": z,
            "total_expand": total_expand,
            "small": small,
            "medium": medium,
            "large": large,
            "total_new_capacity": total_new,
            "current_capacity_012": float(row.get("existing_capacity_0_12", 0.0)),
            "current_capacity_05": float(row.get("existing_capacity_0_5", 0.0)),
            "required_012": float(row.get("min_required_0_12", 0.0)),
            "required_05": float(row.get("min_required_0_5", 0.0))
        })

    return (
        pd.DataFrame(zip_results),
        pd.DataFrame(expanded_facilities),
        pd.DataFrame(new_facilities)
    )



# ============================================
# Main execution
# ============================================

import os

def main():
    print("\nStep 2: Solving optimization problem (procedural version)...")

    data, distances, too_close, facility_data, facility_zip_map, zips, facilities, potential_locations = load_data(
        "processed_data.csv",
        "location_distances.csv",
        "too_close_positions.csv",
        "facility_data.csv",
        "facility_zip_map.csv"
    )

    model = Model("realistic_capacity_planning_procedural")
    model.setParam('OutputFlag', 0)

    x, delta, w, y = create_variables(model, facilities, facility_data, zips, potential_locations)
    add_constraints(model, data, distances, too_close, facility_data, zips, facilities, potential_locations, x, delta, w, y)
    set_objective(model, facilities, facility_data, zips, potential_locations, y, w)
    model.update()

    model = solve_model(model, output_flag=1, mip_gap=0.05, time_limit=3600)

    if model.SolCount > 0:
        zip_df, expand_df, new_df = extract_results(model, data, facility_data, zips, facilities, potential_locations, x, delta, y)

        output_dir = "../Result Data"
        os.makedirs(output_dir, exist_ok=True)

        zip_path = os.path.join(output_dir, "zip_level_summary.csv")
        expand_path = os.path.join(output_dir, "expanded_facilities.csv")
        new_path = os.path.join(output_dir, "new_facilities.csv")

        zip_df.to_csv(zip_path, index=False)
        expand_df.to_csv(expand_path, index=False)
        new_df.to_csv(new_path, index=False)

        print("\n Results saved:")
        print(f"  ZIP summary → {zip_path}")
        print(f"  Expanded facilities → {expand_path}")
        print(f"  New facilities → {new_path}")

        print("\nOptimization results (top ZIPs):")
        print(zip_df.head())
    else:
        print(" No feasible solution found.")

    print(f"\nNumber of ZIP codes: {len(zips)}")
    print(f"Number of facilities: {len(facilities)}")


if __name__ == "__main__":
    main()
