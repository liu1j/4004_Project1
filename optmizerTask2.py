import pandas as pd
from gurobipy import Model, GRB, quicksum

class RealisticCapacityPlanner:
    """解决现实容量扩展和位置问题的优化器（修正版）
    Optimizer for solving realistic capacity expansion and location problems (revised version)"""

    def __init__(self, data_file="processed_data.csv",
                 distance_file="location_distances.csv",
                 too_close_file="too_close_positions.csv"):
        # 数据文件路径 Data file paths
        self.data_file = data_file
        self.distance_file = distance_file
        self.too_close_file = too_close_file
        self.model = None
        self.data = None
        self.distances = None
        self.too_close = None
        self.zips = None
        self.potential_locations = None
        # 设施类型 Facility types
        self.facility_types = ["small", "medium", "large"]
        self.existing_facilities = {}
        # 分段界限（fraction）Segment bounds (fraction)
        self.segment_bounds = [
            (0.0, 0.10),   # 0 - 10%
            (0.10, 0.15),  # 10 - 15%
            (0.15, 0.20)   # 15 - 20%
        ]

    def load_data(self):
        """加载数据并做基础预处理
        Load data and perform basic preprocessing"""
        self.data = pd.read_csv(self.data_file)
        # 确保 zip_code 是整数，使用 unique 列表避免重复
        # Ensure zip_code is integer, use unique list to avoid duplicates
        self.data["zip_code"] = self.data["zip_code"].astype(int)
        self.zips = sorted(self.data["zip_code"].unique())

        # 载入距离与太近位置表
        # Load distance and too-close position tables
        self.distances = pd.read_csv(self.distance_file)
        self.too_close = pd.read_csv(self.too_close_file)

        # 确保这些表的 zip 字段为整数（若存在）
        # Ensure zip fields in these tables are integers (if exist)
        if "zip_code" in self.distances.columns:
            self.distances["zip_code"] = self.distances["zip_code"].astype(int)
        if "zip_code" in self.too_close.columns:
            self.too_close["zip_code"] = self.too_close["zip_code"].astype(int)

        # 构建 potential_locations dict（假设每 zip 在 self.data 中有一行包含潜在位置数量）
        # Build potential_locations dict (assuming each zip has a row in self.data containing the number of potential locations)
        self.potential_locations = {}
        for z in self.zips:
            # 从 self.data 中查找该 zip 的潜在位置数（字段名 'potential_locations' 假定存在）
            # Find the number of potential locations for this zip from self.data (field name 'potential_locations' assumed to exist)
            row = self.data[self.data["zip_code"] == z].iloc[0]
            num_locs = int(row["potential_locations"]) if "potential_locations" in row.index else 0
            self.potential_locations[z] = list(range(num_locs))
            # existing capacity（假设列名为 existing_capacity_0_12）
            # Existing capacity (assuming column name is existing_capacity_0_12)
            self.existing_facilities[z] = int(row.get("existing_capacity_0_12", 0))

    def build_model(self):
        # 如果数据未加载则先加载 If data is not loaded, load it first
        if self.data is None:
            self.load_data()

        self.model = Model("realistic_capacity_planning")
        self.model.setParam('OutputFlag', 0)  # 默认关闭输出，用户可在 solve() 中打开
                                                  # Default to turn off output, user can turn it on in solve()
        self._create_variables()
        self._add_constraints()
        self._set_objective()
        # 确保模型内部变量/约束同步
        # Ensure internal variables/constraints of the model are synchronized
        self.model.update()

    def _create_variables(self):
        """创建变量：x, delta, y，以及线性化用的 w
        Create variables: x, delta, y, and w for linearization"""
        self.x = {}       # continuous expansion amount (slots) for existing facilities (or 0)
                          # 连续扩容量（插槽）用于现有设施（或0）
        self.delta = {}   # binary: which segment chosen for each zip's expansion
                          # 二进制：每个邮编区域扩容选择的段
        self.y = {}       # binary build decisions for each potential location and facility type
                          # 每个潜在位置和设施类型的二进制建设决策
        self.w = {}       # linearization: w[z,k] = x[z] * delta[z,k]
                          # 线性化：w[z,k] = x[z] * delta[z,k]

        for z in self.zips:
            nf = self.existing_facilities.get(z, 0)
            if nf > 0:
                # create continuous expansion var
                # 创建连续扩容变量
                self.x[z] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"expand_{z}")
                # delta binaries for 3 segments
                # 3个段的delta二进制变量
                for k in range(len(self.segment_bounds)):
                    self.delta[z, k] = self.model.addVar(vtype=GRB.BINARY, name=f"delta_{z}_{k}")
                    # linearization vars w
                    # 线性化变量w
                    self.w[z, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{z}_{k}")
            else:
                # 无现有设施：用数值 0 占位（注意后续对 self.x 的判断）
                # No existing facilities: use numeric 0 as placeholder (note subsequent checks on self.x)
                self.x[z] = 0.0
                # 不创建 delta 和 w
                # Do not create delta and w

        # y variables for building new facilities (binary)
        # 建设新设施的y变量（二进制）
        for z in self.zips:
            if z in self.potential_locations and self.potential_locations[z]:
                for l in self.potential_locations[z]:
                    for t in self.facility_types:
                        self.y[z, l, t] = self.model.addVar(vtype=GRB.BINARY, name=f"build_{z}_{l}_{t}")

    def _add_constraints(self):
        """添加约束：分段选择、线性化约束、覆盖约束、距离约束等
        Add constraints: segment selection, linearization constraints, coverage constraints, distance constraints, etc."""
        # For each zip with existing facility(s)
        # 对于每个有现有设施的邮编区域
        for z in self.zips:
            nf = self.existing_facilities.get(z, 0)
            # 获取 expand upper bound（若字段不存在，使用一个较大的备选 M）
            # Get expand upper bound (if field doesn't exist, use a large alternative M)
            expand_ub = None
            dfz = self.data[self.data["zip_code"] == z]
            if not dfz.empty and "expand_upper_bound" in dfz.columns:
                expand_ub = float(dfz["expand_upper_bound"].values[0])
            else:
                # 默认 20% * nf 或一个大值（保守）
                # Default 20% * nf or a large value (conservative)
                expand_ub = 0.2 * nf if nf > 0 else 0.0

            if nf == 0:
                # 没有现有设施，跳过有关 delta 的约束
                # No existing facilities, skip constraints related to delta
                continue

            # (A) 确保选一个分段（保留原逻辑：恰好选1个）
            # (A) Ensure one segment is selected (retain original logic: exactly select 1)
            self.model.addConstr(quicksum(self.delta[z, k] for k in range(len(self.segment_bounds))) == 1,
                                 name=f"delta_sum_{z}")

            # (B) 通过大M把 x 限制到选中的段范围
            # 以及建立 w 的线性化约束： w = x * delta
            # (B) Limit x to the selected segment range via big-M
            # And establish linearization constraints for w: w = x * delta
            M = expand_ub  # big-M
            # 若 M 可能为 0（例如 expand_ub 为 0），令 M 为小正数，避免约束退化
            # If M could be 0 (e.g., expand_ub is 0), set M to a small positive number to avoid constraint degeneration
            if M <= 0:
                M = max(1.0, 0.2 * nf)

            for k, (lower, upper) in enumerate(self.segment_bounds):
                # x >= lower * nf * delta
                self.model.addConstr(self.x[z] >= lower * nf * self.delta[z, k], name=f"expand_lb_{z}_{k}")
                # x <= upper * nf * delta + M * (1 - delta)
                self.model.addConstr(self.x[z] <= upper * nf * self.delta[z, k] + M * (1 - self.delta[z, k]),
                                     name=f"expand_ub_{z}_{k}")
                # w linearization:
                # w 线性化：
                # w <= x
                self.model.addConstr(self.w[z, k] <= self.x[z], name=f"w_le_x_{z}_{k}")
                # w <= M * delta
                self.model.addConstr(self.w[z, k] <= M * self.delta[z, k], name=f"w_le_Md_{z}_{k}")
                # w >= x - M*(1-delta)
                self.model.addConstr(self.w[z, k] >= self.x[z] - M * (1 - self.delta[z, k]), name=f"w_ge_x_minus_M1d_{z}_{k}")
                # w >= 0 (created via var lb)
                # w >= 0 (通过变量下界创建)

            # (C) x cannot exceed expand_ub (global per facility)
            # (C) x不能超过expand_ub（每个设施全局限制）
            self.model.addConstr(self.x[z] <= expand_ub, name=f"expand_limit_{z}")

        # Coverage constraints
        # 覆盖约束
        for z in self.zips:
            # 0-12 coverage
            # 0-12覆盖
            row = self.data[self.data["zip_code"] == z].iloc[0]
            current_012 = float(row.get("existing_capacity_0_12", 0.0))
            req_012 = float(row.get("min_required_0_12", 0.0))

            # new capacity from built facilities
            # 来自建设设施的新容量
            new_capacity = 0
            if z in self.potential_locations and self.potential_locations[z]:
                new_capacity = quicksum(
                    self._get_capacity(t) * self.y[z, l, t]
                    for l in self.potential_locations[z]
                    for t in self.facility_types
                    if (z, l, t) in self.y
                )

            # x[z] may be numeric 0.0 or Gurobi Var
            # x[z]可能是数值0.0或Gurobi变量
            x_var_or_val = self.x[z]
            self.model.addConstr(current_012 + x_var_or_val + new_capacity >= req_012, name=f"cov012_{z}")

            # 0-5 coverage
            # 0-5覆盖
            current_05 = float(row.get("existing_capacity_0_5", 0.0))
            req_05 = float(row.get("min_required_0_5", 0.0))
            new_05 = 0
            if z in self.potential_locations and self.potential_locations[z]:
                new_05 = quicksum(
                    self._get_05_capacity(t) * self.y[z, l, t]
                    for l in self.potential_locations[z]
                    for t in self.facility_types
                    if (z, l, t) in self.y
                )
            # 将扩容 x 计入 0-5 覆盖（保守做法）
            # Count expansion x in 0-5 coverage (conservative approach)
            self.model.addConstr(current_05 + x_var_or_val + new_05 >= req_05, name=f"cov05_{z}")

        # Distance constraints: new-new (from distances file) and new-existing (from too_close file)
        # 距离约束：新建-新建（来自距离文件）和新建-现有（来自too_close文件）
        # new-new
        if self.distances is not None and not self.distances.empty:
            # 假设 distances 有列: zip_code, loc1_id, loc2_id, too_close (True/False)
            # Assume distances has columns: zip_code, loc1_id, loc2_id, too_close (True/False)
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
                if z not in self.potential_locations:
                    continue
                if loc not in self.potential_locations[z]:
                    continue
                for t in self.facility_types:
                    if (z, loc, t) in self.y:
                        self.model.addConstr(self.y[z, loc, t] == 0, name=f"distance_exist_{z}_{loc}_{t}")

    def _set_objective(self):
        """设置线性化后的目标函数：build_cost + equip_cost + expand_cost(通过 w)
        Set the linearized objective function: build_cost + equip_cost + expand_cost (via w)"""
        # Build cost
        # 建设成本
        build_cost = quicksum(
            self._get_build_cost(t) * self.y[z, l, t]
            for z in self.zips
            if z in self.potential_locations and self.potential_locations[z]
            for l in self.potential_locations[z]
            for t in self.facility_types
            if (z, l, t) in self.y
        )

        # Equipment cost for 0-5 slots in new facilities: $100 per 0-5 slot
        # 新设施建设中0-5年龄段插槽的设备成本：每个0-5插槽$100
        equip_cost = quicksum(
            100.0 * self._get_05_capacity(t) * self.y[z, l, t]
            for z in self.zips
            if z in self.potential_locations and self.potential_locations[z]
            for l in self.potential_locations[z]
            for t in self.facility_types
            if (z, l, t) in self.y
        )

        # Expansion cost via w (linearized)
        # 通过w的扩容成本（线性化）
        expand_cost_terms = []
        for z in self.zips:
            nf = self.existing_facilities.get(z, 0)
            if nf == 0:
                continue
            for k in range(len(self.segment_bounds)):
                # cost factor per segment: (20,000 + factor * nf) * (x / nf)
                # 每段的成本因子：(20,000 + factor * nf) * (x / nf)
                cost_factor = 20000.0 + self._get_cost_factor(k) * nf
                # term = cost_factor * w[z,k] / nf
                expand_cost_terms.append((cost_factor / float(nf)) * self.w[z, k])

        expand_cost = quicksum(expand_cost_terms) if expand_cost_terms else 0.0

        # Total objective
        # 总目标函数
        self.model.setObjective(build_cost + equip_cost + expand_cost, GRB.MINIMIZE)

    def _get_cost_factor(self, segment_idx):
        """分段容量系数（对应 PDF 中 200, 400, 1000）"""
        cost_factors = [200, 400, 1000]
        return cost_factors[segment_idx]

    def _get_capacity(self, facility_type):
        """总容量（0-12? or overall slots）——这里返回表中小/中/大对应的总容量（用于 0-12 覆盖估计）"""
        if facility_type == "small":
            return 100
        elif facility_type == "medium":
            return 200
        else:
            return 400

    def _get_05_capacity(self, facility_type):
        """0-5 年龄段专用容量（表格给定: small 50, medium 100, large 200）"""
        if facility_type == "small":
            return 50
        elif facility_type == "medium":
            return 100
        else:
            return 200

    def _get_build_cost(self, facility_type):
        # 建设成本 Facility costs
        if facility_type == "small":
            return 65000.0
        elif facility_type == "medium":
            return 95000.0
        else:
            return 115000.0

    def solve(self, output_flag=1, mip_gap=0.05, time_limit=3600):
        """求解模型并在非最优但有可行解时仍提取当前解
        Solve the model and extract current solution even when non-optimal but feasible"""
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
            # 如果时间到但有可行解，则返回该解
            # If time limit reached but there is a feasible solution, return that solution
            if self.model.SolCount > 0:
                print("Time limit reached — returning incumbent feasible solution.")
                # 时间到—返回现有可行解
                return self._extract_results()
            else:
                print("Time limit reached — no feasible solution found.")
                # 时间到—未找到可行解
                return None
        elif status == GRB.INFEASIBLE:
            print("Model infeasible.")
            # 模型不可行
            return None
        else:
            # 其他状态（可行解但非 optimal）
            # Other statuses (feasible solution but not optimal)
            if self.model.SolCount > 0:
                print(f"Solver status {status} — returning incumbent solution.")
                # 求解器状态{status}—返回现有解
                return self._extract_results()
            print(f"Solver ended with status {status}. No solution returned.")
            # 求解器以状态{status}结束。未返回解
            return None

    def _extract_results(self):
        """从求解器提取决策变量值并构造 DataFrame
        Extract decision variable values from solver and construct DataFrame"""
        results = []
        for z in self.zips:
            row = self.data[self.data["zip_code"] == z].iloc[0]
            # 提取 x 值（可能是数字或 Gurobi Var）
            # Extract x value (could be numeric or Gurobi Var)
            x_val = 0.0
            if hasattr(self.x[z], "X"):
                x_val = float(self.x[z].X)
            else:
                # numeric (例如 0.0)
                # numeric (e.g., 0.0)
                x_val = float(self.x[z])

            # 找到哪个 segment 被选中
            # Find which segment was selected
            seg = "N/A"
            for k in range(len(self.segment_bounds)):
                if (z, k) in self.delta and hasattr(self.delta[z, k], "X") and self.delta[z, k].X > 0.5:
                    seg = f"{int(self.segment_bounds[k][0]*100)}%-{int(self.segment_bounds[k][1]*100)}%"
                    break

            # 新建设施计数
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
                "expand": x_val,
                "expand_segment": seg,
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

    def save_results(self, filename="realistic_capacity_solution.csv"):
        # 求解结果 Solve results
        results = self.solve()
        if results is not None:
            results.to_csv(filename, index=False)
            print(f"结果已保存至 {filename}")
            # Results saved to {filename}
            print("\n关键统计信息:")
            # Key statistics:
            print(f"总扩容量: {results['expand'].sum():,.0f} 托位")
            # Total expansion capacity: {results['expand'].sum():,.0f} slots
            print(f"新建设施数量: {results[['small','medium','large']].sum().sum():,.0f}")
            # Number of new facilities: {results[['small','medium','large']].sum().sum():,.0f}
            # 扩容区间分布 Distribution of expansion intervals
            for seg in ["0%-10%", "10%-15%", "15%-20%"]:
                count = results[results['expand_segment'] == seg].shape[0]
                print(f"  {seg}: {count} 个区域")
        else:
            print("没有可保存的结果。")
            # No results to save.

if __name__ == "__main__":
    print("\nStep 2: Solving optimization problem...")
    # 第2步：求解优化问题...
    planner = RealisticCapacityPlanner()
    # 可选地调整 solver 输出/参数
    # Optionally adjust solver output/parameters
    res = planner.solve(output_flag=1, mip_gap=0.05, time_limit=3600)
    if res is not None:
        print("\nOptimization results (top rows):")
        # 优化结果（前几行）:
        print(res.head())
        planner.save_results()
    print("\nKey Model Information:")
    # 关键模型信息:
    print(f"Number of ZIP codes: {len(planner.zips)}")
    # 邮编区域数量: {len(planner.zips)}