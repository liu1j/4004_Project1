import pandas as pd
from gurobipy import Model, GRB, quicksum

class RealisticCapacityPlanner:
    """Optimizer for solving realistic capacity expansion and location problems (per facility expansion)"""
    def __init__(self, data_file="processed_data.csv",
                 distance_file="location_distances.csv",
                 too_close_file="too_close_positions.csv",
                 facility_data_file="facility_data.csv",
                 facility_zip_map_file="facility_zip_map.csv"):
        # Data file paths
        self.data_file = data_file
        self.distance_file = distance_file
        self.too_close_file = too_close_file
        self.facility_data_file = facility_data_file
        self.facility_zip_map_file = facility_zip_map_file
        self.model = None
        self.data = None
        self.distances = None
        self.too_close = None
        self.facility_data = None
        self.facility_zip_map = None
        self.zips = None
        self.facilities = None  # Store all facility IDs
        self.potential_locations = None
        # Facility types
        self.facility_types = ["small", "medium", "large"]
        # Segment bounds (fraction)
        self.segment_bounds = [
            (0.0, 0.10),   # 0 - 10%
            (0.10, 0.15),  # 10 - 15%
            (0.15, 0.20)   # 15 - 20%
        ]
    
    def load_data(self):
        """Load data and perform basic preprocessing"""
        self.data = pd.read_csv(self.data_file)
        self.facility_data = pd.read_csv(self.facility_data_file)
        self.facility_zip_map = pd.read_csv(self.facility_zip_map_file)
        
        # Ensure zip_code is integer
        self.data["zip_code"] = self.data["zip_code"].astype(int)
        self.facility_data["zip_code"] = self.facility_data["zip_code"].astype(int)
        self.facility_zip_map["zip_code"] = self.facility_zip_map["zip_code"].astype(int)
        
        # Get all zip codes
        self.zips = sorted(self.data["zip_code"].unique())
        
        # Get all facility IDs
        self.facilities = self.facility_data["facility_id"].tolist()
        
        # Load distance and too-close position tables
        self.distances = pd.read_csv(self.distance_file)
        self.too_close = pd.read_csv(self.too_close_file)
        
        # Ensure zip fields in these tables are integers (if exist)
        if "zip_code" in self.distances.columns:
            self.distances["zip_code"] = self.distances["zip_code"].astype(int)
        if "zip_code" in self.too_close.columns:
            self.too_close["zip_code"] = self.too_close["zip_code"].astype(int)
        
        # Build potential_locations dict
        self.potential_locations = {}
        for z in self.zips:
            # Find the number of potential locations for this zip from self.data
            row = self.data[self.data["zip_code"] == z].iloc[0]
            num_locs = int(row["potential_locations"]) if "potential_locations" in row.index else 0
            self.potential_locations[z] = list(range(num_locs))

    def build_model(self):
        # If data is not loaded, load it first
        if self.data is None:
            self.load_data()
        self.model = Model("realistic_capacity_planning_per_facility")
        self.model.setParam('OutputFlag', 0)  # Default to turn off output, user can turn it on in solve()
        self._create_variables()
        self._add_constraints()
        self._set_objective()
        # Ensure internal variables/constraints of the model are synchronized
        self.model.update()

    def _create_variables(self):
        """Create variables: x, delta, y, and w for linearization"""
        self.x = {}       # continuous expansion amount (slots) for each facility
        self.delta = {}   # binary: which segment chosen for each facility's expansion
        self.w = {}       # linearization: w[f,k] = x[f] * delta[f,k]
        
        # Create expansion decision variables for each facility
        for facility_id in self.facilities:
            # Get initial capacity of this facility
            facility_row = self.facility_data[self.facility_data["facility_id"] == facility_id]
            if facility_row.empty:
                continue
            initial_capacity = float(facility_row["initial_capacity_0_12"].values[0])
            
            # Only create expansion variables when initial capacity > 0
            if initial_capacity > 0:
                # Create continuous expansion variable
                self.x[facility_id] = self.model.addVar(vtype=GRB.INTEGER, lb=0.0, 
                                                      name=f"expand_{facility_id}")
                
                # Create delta binary variables for 3 segments
                for k in range(len(self.segment_bounds)):
                    self.delta[facility_id, k] = self.model.addVar(vtype=GRB.BINARY, 
                                                                 name=f"delta_{facility_id}_{k}")
                    # Create linearization variable w
                    self.w[facility_id, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, 
                                                             name=f"w_{facility_id}_{k}")
            else:
                # Initial capacity is 0, use numeric 0 as placeholder
                self.x[facility_id] = 0.0
        
        # y variables for building new facilities (binary)
        self.y = {}
        for z in self.zips:
            if z in self.potential_locations and self.potential_locations[z]:
                for l in self.potential_locations[z]:
                    for t in self.facility_types:
                        self.y[z, l, t] = self.model.addVar(vtype=GRB.BINARY, name=f"build_{z}_{l}_{t}")

    def _add_constraints(self):
        """Add constraints: segment selection, linearization constraints, coverage constraints, distance constraints, etc."""
        # For each facility
        for facility_id in self.facilities:
            # Get initial capacity of this facility
            facility_row = self.facility_data[self.facility_data["facility_id"] == facility_id]
            if facility_row.empty:
                continue
            initial_capacity = float(facility_row["initial_capacity_0_12"].values[0])
            
            # Only add expansion constraints when initial capacity > 0
            if initial_capacity <= 0:
                continue
                
            # (A) Ensure one segment is selected
            self.model.addConstr(quicksum(self.delta[facility_id, k] for k in range(len(self.segment_bounds))) == 1,
                                 name=f"delta_sum_{facility_id}")
            
            # (B) Limit x to the selected segment range via big-M
            # And establish linearization constraints for w: w = x * delta
            M = initial_capacity * 0.2  # Big-M value is 20% of initial capacity
            # If M could be 0, set M to a small positive number
            if M <= 0:
                M = 1.0
                
            for k, (lower, upper) in enumerate(self.segment_bounds):
                # x >= lower * initial_capacity * delta
                self.model.addConstr(self.x[facility_id] >= lower * initial_capacity * self.delta[facility_id, k], 
                                     name=f"expand_lb_{facility_id}_{k}")
                # x <= upper * initial_capacity * delta + M * (1 - delta)
                self.model.addConstr(self.x[facility_id] <= upper * initial_capacity * self.delta[facility_id, k] + 
                                     M * (1 - self.delta[facility_id, k]),
                                     name=f"expand_ub_{facility_id}_{k}")
                # w linearization:
                # w <= x
                self.model.addConstr(self.w[facility_id, k] <= self.x[facility_id], 
                                     name=f"w_le_x_{facility_id}_{k}")
                # w <= M * delta
                self.model.addConstr(self.w[facility_id, k] <= M * self.delta[facility_id, k], 
                                     name=f"w_le_Md_{facility_id}_{k}")
                # w >= x - M*(1-delta)
                self.model.addConstr(self.w[facility_id, k] >= self.x[facility_id] - 
                                     M * (1 - self.delta[facility_id, k]), 
                                     name=f"w_ge_x_minus_M1d_{facility_id}_{k}")
                # w >= 0 (created via var lb)
        
        # Coverage constraints (by zip code)
        for z in self.zips:
            # Get data for this zip code
            zip_row = self.data[self.data["zip_code"] == z].iloc[0]
            current_012 = float(zip_row.get("existing_capacity_0_12", 0.0))
            req_012 = float(zip_row.get("min_required_0_12", 0.0))
            
            # Calculate total expansion from all facilities
            total_expansion = 0.0
            facilities_in_zip = self.facility_data[self.facility_data["zip_code"] == z]["facility_id"].tolist()
            for facility_id in facilities_in_zip:
                if facility_id in self.x:
                    # x[facility_id] could be a variable or a numeric value
                    if hasattr(self.x[facility_id], "X"):
                        total_expansion += self.x[facility_id]
                    else:
                        total_expansion += self.x[facility_id]
            
            # Capacity from new facilities
            new_capacity = 0
            if z in self.potential_locations and self.potential_locations[z]:
                new_capacity = quicksum(
                    self._get_capacity(t) * self.y[z, l, t]
                    for l in self.potential_locations[z]
                    for t in self.facility_types
                    if (z, l, t) in self.y
                )
            
            # 0-12 coverage constraint
            self.model.addConstr(current_012 + total_expansion + new_capacity >= req_012, 
                                 name=f"cov012_{z}")
            
            # 0-5 coverage constraint
            current_05 = float(zip_row.get("existing_capacity_0_5", 0.0))
            req_05 = float(zip_row.get("min_required_0_5", 0.0))
            new_05 = 0
            if z in self.potential_locations and self.potential_locations[z]:
                new_05 = quicksum(
                    self._get_05_capacity(t) * self.y[z, l, t]
                    for l in self.potential_locations[z]
                    for t in self.facility_types
                    if (z, l, t) in self.y
                )
            
            # Calculate total expansion for 0-5 (conservative approach)
            total_expansion_05 = 0.0
            for facility_id in facilities_in_zip:
                if facility_id in self.x:
                    # 0-5 expansion is same as total expansion (conservative assumption)
                    if hasattr(self.x[facility_id], "X"):
                        total_expansion_05 += self.x[facility_id]
                    else:
                        total_expansion_05 += self.x[facility_id]
            
            self.model.addConstr(current_05 + total_expansion_05 + new_05 >= req_05, 
                                 name=f"cov05_{z}")
        
        # Distance constraints: new-new and new-existing
        # new-new
        if self.distances is not None and not self.distances.empty:
            if "too_close" in self.distances.columns:
                close_pairs = self.distances[self.distances["too_close"] == True]
            else:
                close_pairs = self.distances
            for _, row in close_pairs.iterrows():
                z = int(row["zip_code"])
                loc1 = int(row["loc1_id"])
                loc2 = int(row["loc2_id"])
                if z not in self.potential_locations:
                    continue
                if loc1 not in self.potential_locations[z] or loc2 not in self.potential_locations[z]:
                    continue
                for t in self.facility_types:
                    if (z, loc1, t) in self.y and (z, loc2, t) in self.y:
                        self.model.addConstr(self.y[z, loc1, t] + self.y[z, loc2, t] <= 1,
                                             name=f"distance_new_{z}_{loc1}_{loc2}_{t}")
        
        # new-existing
        if self.too_close is not None and not self.too_close.empty:
            for _, row in self.too_close.iterrows():
                z = int(row["zip_code"])
                loc = int(row["location_id"])
                facility_id = int(row["facility_id"])
                if z not in self.potential_locations:
                    continue
                if loc not in self.potential_locations[z]:
                    continue
                for t in self.facility_types:
                    if (z, loc, t) in self.y:
                        self.model.addConstr(self.y[z, loc, t] == 0, 
                                            name=f"distance_exist_{z}_{loc}_{facility_id}_{t}")
                        
        # New constraint: at most one facility type can be built at each potential location (z, l)
        for z in self.zips:
            if z in self.potential_locations and self.potential_locations[z]:
                for l in self.potential_locations[z]:
                    self.model.addConstr(
                        quicksum(self.y[z, l, t] for t in self.facility_types if (z, l, t) in self.y) <= 1,
                        name=f"one_facility_per_location_{z}_{l}"
                    )

    def _set_objective(self):
        """Set the linearized objective function: build_cost + equip_cost + expand_cost (via w)"""
        # Build cost
        build_cost = quicksum(
            self._get_build_cost(t) * self.y[z, l, t]
            for z in self.zips
            if z in self.potential_locations and self.potential_locations[z]
            for l in self.potential_locations[z]
            for t in self.facility_types
            if (z, l, t) in self.y
        )
        
        # Equipment cost for 0-5 slots in new facilities: $100 per 0-5 slot
        equip_cost = quicksum(
            100.0 * self._get_05_capacity(t) * self.y[z, l, t]
            for z in self.zips
            if z in self.potential_locations and self.potential_locations[z]
            for l in self.potential_locations[z]
            for t in self.facility_types
            if (z, l, t) in self.y
        )
        
        # Expansion cost via w (linearized)
        expand_cost_terms = []
        for facility_id in self.facilities:
            # Get initial capacity of this facility
            facility_row = self.facility_data[self.facility_data["facility_id"] == facility_id]
            if facility_row.empty:
                continue
            initial_capacity = float(facility_row["initial_capacity_0_12"].values[0])
            
            # Only calculate expansion cost when initial capacity > 0
            if initial_capacity <= 0:
                continue
                
            for k in range(len(self.segment_bounds)):
                # cost factor per segment: (20,000 + factor * initial_capacity) * (x / initial_capacity)
                cost_factor = 20000.0 + self._get_cost_factor(k) * initial_capacity
                # term = cost_factor * w[facility_id,k] / initial_capacity
                expand_cost_terms.append((cost_factor / float(initial_capacity)) * self.w[facility_id, k])
        
        expand_cost = quicksum(expand_cost_terms) if expand_cost_terms else 0.0
        
        # Total objective
        self.model.setObjective(build_cost + equip_cost + expand_cost, GRB.MINIMIZE)

    def _get_cost_factor(self, segment_idx):
        """Cost factors for segments (corresponding to 200, 400, 1000 in PDF)"""
        cost_factors = [200, 400, 1000]
        return cost_factors[segment_idx]

    def _get_capacity(self, facility_type):
        """Total capacity (0-12)"""
        if facility_type == "small":
            return 100
        elif facility_type == "medium":
            return 200
        else:
            return 400

    def _get_05_capacity(self, facility_type):
        """Dedicated capacity for 0-5 age group"""
        if facility_type == "small":
            return 50
        elif facility_type == "medium":
            return 100
        else:
            return 200

    def _get_build_cost(self, facility_type):
        """Construction cost"""
        if facility_type == "small":
            return 65000.0
        elif facility_type == "medium":
            return 95000.0
        else:
            return 115000.0

    def solve(self, output_flag=1, mip_gap=0.05, time_limit=3600):
        """Solve the model and extract current solution even when non-optimal but feasible"""
        if self.model is None:
            self.build_model()
        self.model.setParam('OutputFlag', int(output_flag))
        self.model.setParam('MIPGap', float(mip_gap))
        self.model.setParam('TimeLimit', float(time_limit))
        self.model.optimize()
        status = self.model.status
        if status == GRB.OPTIMAL:
            return self._extract_results()
        elif status == GRB.TIME_LIMIT:
            # If time limit reached but there is a feasible solution, return that solution
            if self.model.SolCount > 0:
                print("Time limit reached — returning incumbent feasible solution.")
                return self._extract_results()
            else:
                print("Time limit reached — no feasible solution found.")
                return None
        elif status == GRB.INFEASIBLE:
            print("Model infeasible.")
            return None
        else:
            # Other statuses (feasible solution but not optimal)
            if self.model.SolCount > 0:
                print(f"Solver status {status} — returning incumbent solution.")
                return self._extract_results()
            print(f"Solver ended with status {status}. No solution returned.")
            return None

    def _extract_results(self):
        """Extract decision variable values from solver and construct DataFrame"""
        # First, get overall information for each zip code
        results = []
        for z in self.zips:
            row = self.data[self.data["zip_code"] == z].iloc[0]
            # Extract expansion information for all facilities in this zip code
            facilities_in_zip = self.facility_data[self.facility_data["zip_code"] == z]
            total_expand = 0.0
            expand_details = []
            
            for _, facility in facilities_in_zip.iterrows():
                facility_id = facility["facility_id"]
                # Extract x value
                x_val = 0.0
                if facility_id in self.x and hasattr(self.x[facility_id], "X"):
                    x_val = float(self.x[facility_id].X)
                elif facility_id in self.x:
                    x_val = float(self.x[facility_id])
                
                # Accumulate total expansion
                total_expand += x_val
                
                # Find selected segment
                seg = "N/A"
                for k in range(len(self.segment_bounds)):
                    if (facility_id, k) in self.delta and hasattr(self.delta[facility_id, k], "X") and self.delta[facility_id, k].X > 0.5:
                        seg = f"{int(self.segment_bounds[k][0]*100)}%-{int(self.segment_bounds[k][1]*100)}%"
                        break
                
                expand_details.append({
                    "facility_id": facility_id,
                    "expand": x_val,
                    "expand_segment": seg,
                    "initial_capacity": float(facility["initial_capacity_0_12"])
                })
            
            # Count of new facilities
            small = medium = large = 0
            if z in self.potential_locations and self.potential_locations[z]:
                for l in self.potential_locations[z]:
                    for t in self.facility_types:
                        if (z, l, t) in self.y and hasattr(self.y[z, l, t], "X") and self.y[z, l, t].X > 0.5:
                            if t == "small":
                                small += 1
                            elif t == "medium":
                                medium += 1
                            else:
                                large += 1
            
            total_new = small * self._get_capacity("small") + medium * self._get_capacity("medium") + large * self._get_capacity("large")
            results.append({
                "zip": z,
                "total_expand": total_expand,
                "expand_details": expand_details,
                "small": small,
                "medium": medium,
                "large": large,
                "total_new_capacity": total_new,
                "current_capacity_012": float(row.get("existing_capacity_0_12", 0.0)),
                "current_capacity_05": float(row.get("existing_capacity_0_5", 0.0)),
                "required_012": float(row.get("min_required_0_12", 0.0)),
                "required_05": float(row.get("min_required_0_5", 0.0))
            })
        
        return pd.DataFrame(results)

    def save_results(self, 
                     expansion_file="expansion_results.csv",
                     new_facility_file="new_facility_results.csv"):
        """Save results to two CSV files:
        - expansion_file: Details of expansion for existing facilities
        - new_facility_file: Locations and types of new facilities
        """
        results = self.solve()
        if results is None:
            print("No results to save.")
            return

        # ====== 1. Save expansion information ======
        expansion_data = []
        for _, row in results.iterrows():
            zip_code = row['zip']
            for facility in row['expand_details']:
                # Only save facilities with actual expansion (or decisions), even if expand=0 (as segment might be selected)
                expansion_data.append({
                    "zip_code": zip_code,
                    "facility_id": facility["facility_id"],
                    "initial_capacity_0_12": facility["initial_capacity"],
                    "expand_slots": facility["expand"],
                    "expand_segment": facility["expand_segment"]
                })

        if expansion_data:
            exp_df = pd.DataFrame(expansion_data)
            exp_df.to_csv(expansion_file, index=False)
            print(f"Expansion details for existing facilities saved to {expansion_file}")
        else:
            print("No expansion details to save.")

        # ====== 2. Save new facility information ======
        new_facility_data = []
        for z in self.zips:
            if z not in self.potential_locations or not self.potential_locations[z]:
                continue
            for l in self.potential_locations[z]:
                for t in self.facility_types:
                    if (z, l, t) in self.y:
                        var = self.y[z, l, t]
                        if hasattr(var, "X") and var.X > 0.5:
                            new_facility_data.append({
                                "zip_code": z,
                                "location_id": l,
                                "facility_type": t,
                                "capacity_0_12": self._get_capacity(t),
                                "capacity_0_5": self._get_05_capacity(t)
                            })

        if new_facility_data:
            new_df = pd.DataFrame(new_facility_data)
            new_df.to_csv(new_facility_file, index=False)
            print(f"New facility information saved to {new_facility_file}")
        else:
            print("No new facility information to save.")

        # ====== Print summary statistics ======
        total_expand = results['total_expand'].sum()
        total_new_facilities = results[['small', 'medium', 'large']].sum().sum()
        print("\nKey Statistics:")
        print(f"Total expanded capacity: {total_expand:,.0f} slots")
        print(f"Number of new facilities: {total_new_facilities:,.0f}")

        # Expansion segment distribution
        all_segments = []
        for _, row in results.iterrows():
            for facility in row['expand_details']:
                if facility['expand_segment'] != 'N/A':
                    all_segments.append(facility['expand_segment'])
        for seg in ["0%-10%", "10%-15%", "15%-20%"]:
            count = all_segments.count(seg)
            print(f"  {seg}: {count} facilities")

if __name__ == "__main__":
    print("\nStep 2: Solving optimization problem (per facility expansion)...")
    # Step 2: Solve optimization problem (per facility expansion)...
    planner = RealisticCapacityPlanner()
    # Optionally adjust solver output/parameters
    res = planner.solve(output_flag=1, mip_gap=0.05, time_limit=3600)
    if res is not None:
        print("\nOptimization results (top rows):")
        print(res.head())
        planner.save_results()
    print("\nKey Model Information:")
    print(f"Number of ZIP codes: {len(planner.zips)}")
    print(f"Number of facilities: {len(planner.facilities)}")
