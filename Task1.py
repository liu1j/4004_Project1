import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum, Env

class ChildCareDesertEliminator:
    """Child care desert elimination optimization model"""
    
    def __init__(self):
        # 初始化数据容器 (Initialize data containers)
        self.df_zip = None
        self.ZIPS = None
        self.existing_0_12 = {}
        self.existing_0_5 = {}
        self.req_012 = {}
        self.req_05 = {}
        self.expand_upper_bound = {}
        
        # 设施类型参数 (Facility type parameters)
        self.types = ["small", "medium", "large"]
        self.capacity = {"small": 100, "medium": 200, "large": 400}
        self.capacity_05 = {"small": 50, "medium": 100, "large": 200}
        self.cost_build = {"small": 65000, "medium": 95000, "large": 115000}
        
        # 模型对象 (Model objects)
        self.model = None
        self.variables = {}
    
    def load_and_preprocess_data(self):
        """加载并预处理所有数据 (Load and preprocess all data)"""
        # 读取五个数据文件 (Read five data files)
        df_fac = pd.read_csv("child_care_regulated.csv")
        df_inc = pd.read_csv("avg_individual_income.csv")
        df_pop = pd.read_csv("population.csv")
        df_emp = pd.read_csv("employment_rate.csv")
        
        # 处理ZIP码 (Process ZIP codes)
        self._process_zip_codes(df_fac, df_inc, df_pop, df_emp)
        
        # 计算容量 (Calculate capacities)
        self._calculate_capacities(df_fac)
        
        # 计算人口 (Calculate population)
        self._calculate_population(df_pop)
        
        # 合并数据 - 修复：传递df_fac和df_pop参数 (Merge data - fix: pass df_fac and df_pop parameters)
        self._merge_data(df_fac, df_inc, df_pop, df_emp)
        
        # 计算需求 (Calculate requirements)
        self._calculate_requirements()
        
        # 准备ZIP码列表 (Prepare ZIP code list)
        self.ZIPS = list(self.df_zip["zip_code"].unique())
        
        # 创建参数字典 (Create parameter dictionaries)
        self._create_parameter_dicts()
        
        # 计算扩容上限 (Calculate expansion upper bound)
        self._calculate_expand_upper_bound()

    def _process_zip_codes(self, df_fac, df_inc, df_pop, df_emp):
        """处理ZIP码格式 (Process ZIP code format)"""
        for df in [df_fac, df_inc, df_pop, df_emp]:
            if "zip_code" in df.columns:
                df.loc[df["zip_code"] >= 100000, "zip_code"] = df["zip_code"] // 10000
            elif "ZIP code" in df.columns:
                df.loc[df["ZIP code"] >= 100000, "ZIP code"] = df["ZIP code"] // 10000
            elif "zipcode" in df.columns:
                df.loc[df["zipcode"] >= 100000, "zipcode"] = df["zipcode"] // 10000

    def _calculate_capacities(self, df_fac):
        """计算托儿机构容量 (Calculate childcare facility capacity)"""
        df_fac["existing_capacity_0_12"] = (
            df_fac[["infant_capacity", "toddler_capacity", "preschool_capacity"]].sum(axis=1)
            + (5/12) * df_fac["children_capacity"]
        )
        df_fac["existing_capacity_0_5"] = df_fac[["infant_capacity", "toddler_capacity"]].sum(axis=1)

    def _calculate_population(self, df_pop):
        """计算人口统计数据 (Calculate population statistics)"""
        df_pop["pop_0_5"] = df_pop["-5"]
        df_pop["10-12"] = df_pop["10-14"] * (3/5)
        df_pop["pop_0_12"] = df_pop[["-5", "5-9", "10-12"]].sum(axis=1)

    def _merge_data(self, df_fac, df_inc, df_pop, df_emp):
        """合并所有数据集 (Merge all datasets)"""
        cap_by_zip = (
            df_fac.groupby("zip_code")[["existing_capacity_0_12", "existing_capacity_0_5"]]
            .sum()
            .reset_index()
        )
        
        pop_by_zip = df_pop[["zipcode", "pop_0_5", "pop_0_12"]].rename(columns={"zipcode": "zip_code"})
        inc_by_zip = df_inc.rename(columns={"ZIP code": "zip_code"})
        emp_by_zip = df_emp.rename(columns={"zipcode": "zip_code"})
        
        self.df_zip = (
            cap_by_zip
            .merge(pop_by_zip, on="zip_code", how="outer")
            .merge(inc_by_zip, on="zip_code", how="outer")
            .merge(emp_by_zip, on="zip_code", how="outer")
        )
        
        # 删除关键字段缺失的行 (Remove rows with missing key fields)
        self.df_zip = self.df_zip.dropna(subset=["existing_capacity_0_12", "pop_0_12", "pop_0_5"])

    def _calculate_requirements(self):
        """计算最低需求 (Calculate minimum requirements)"""
        self.df_zip["high_demand"] = (self.df_zip["average income"] <= 60000) | (self.df_zip["employment rate"] >= 0.6)
        self.df_zip["min_required_0_12"] = self.df_zip.apply(
            lambda r: 0.5 * r["pop_0_12"] if r["high_demand"] else (1/3) * r["pop_0_12"],
            axis=1
        )
        self.df_zip["min_required_0_5"] = (2/3) * self.df_zip["pop_0_5"]

    def _create_parameter_dicts(self):
        """创建参数字典供模型使用 (Create parameter dictionaries for model use)"""
        self.existing_0_12 = self.df_zip.set_index("zip_code")["existing_capacity_0_12"].to_dict()
        self.existing_0_5 = self.df_zip.set_index("zip_code")["existing_capacity_0_5"].to_dict()
        self.req_012 = self.df_zip.set_index("zip_code")["min_required_0_12"].to_dict()
        self.req_05 = self.df_zip.set_index("zip_code")["min_required_0_5"].to_dict()

    def _calculate_expand_upper_bound(self):
        """计算每个ZIP码的扩容上限 (Calculate expansion upper bound for each ZIP code)"""
        self.expand_upper_bound = {}
        for z in self.ZIPS:
            current_capacity = self.existing_0_12[z]
            self.expand_upper_bound[z] = min(1.2 * current_capacity, 500)

    def build_model(self):
        """构建优化模型 (Build optimization model)"""
        self.model = Model("child_desert_elimination")
        
        # 创建变量 (Create variables)
        self._create_variables()
        
        # 添加约束 (Add constraints)
        self._add_constraints()
        
        # 设置目标函数 (Set objective function)
        self._set_objective()

    def _create_variables(self):
        """创建决策变量 (Create decision variables)"""
        # y[z,t]：在ZIP z新建类型t的设施数量 (Number of new facilities of type t built in ZIP z)
        self.variables["y"] = self.model.addVars(
            self.ZIPS, self.types, 
            vtype=GRB.INTEGER, 
            name="build"
        )
        
        # z05[z,t]：在ZIP z的类型t设施中分配的0-5岁托位数 (Number of slots for ages 0-5 assigned to facility type t in ZIP z)
        self.variables["z05"] = self.model.addVars(
            self.ZIPS, self.types, 
            vtype=GRB.INTEGER, 
            name="slots05"
        )
        
        # x[z]：在ZIP z对现有设施的扩容数量 (Number of expansions to existing facilities in ZIP z)
        self.variables["x"] = self.model.addVars(
            self.ZIPS, 
            vtype=GRB.INTEGER, 
            name="expand"
        )
        
        # big_expansion[z]：标记扩容是否≥100%的二元变量 (Binary variable indicating if expansion is ≥100%)
        self.variables["big_expansion"] = self.model.addVars(
            self.ZIPS, 
            vtype=GRB.BINARY, 
            name="big_expansion"
        )

    def _add_constraints(self):
        """添加所有约束 (Add all constraints)"""
        # 1. 0-12岁托位覆盖约束 (0-12 age group coverage constraint)
        for z in self.ZIPS:
            self.model.addConstr(
                self.existing_0_12[z] + self.variables["x"][z] + 
                quicksum(self.capacity[t] * self.variables["y"][z, t] for t in self.types) >= self.req_012[z],
                name=f"cov012_{z}"
            )
        
        # 2. 0-5岁托位覆盖约束 (0-5 age group coverage constraint)
        for z in self.ZIPS:
            self.model.addConstr(
                self.existing_0_5[z] + quicksum(self.variables["z05"][z, t] for t in self.types) >= self.req_05[z],
                name=f"cov05_{z}"
            )
        
        # 3. 每个设施的0-5托位不能超过专用容量上限 (0-5 slots per facility cannot exceed dedicated capacity limit)
        for z in self.ZIPS:
            for t in self.types:
                self.model.addConstr(
                    self.variables["z05"][z, t] <= self.capacity_05[t] * self.variables["y"][z, t],
                    name=f"limit05_{z}_{t}"
                )
        
        # 4. 扩容上限约束 (Expansion upper bound constraint)
        for z in self.ZIPS:
            self.model.addConstr(
                self.variables["x"][z] <= self.expand_upper_bound[z],
                name=f"expand_limit_{z}"
            )
        
        # 5. 防止对无现有设施的区域扩容 (Prevent expansion in areas with no existing facilities)
        for z in self.ZIPS:
            if self.existing_0_12[z] == 0:
                self.model.addConstr(
                    self.variables["x"][z] == 0,
                    name=f"no_expand_{z}"
                )
        
        # 6. 大规模扩容条件约束 (Large expansion condition constraints)
        for z in self.ZIPS:
            if self.existing_0_12[z] > 0:
                # 如果big_expansion=1，则x >= C (If big_expansion=1, then x >= C)
                self.model.addConstr(
                    self.variables["x"][z] >= self.existing_0_12[z] * self.variables["big_expansion"][z],
                    name=f"big_expansion_1_{z}"
                )
                
                # 如果big_expansion=0，则x < C (If big_expansion=0, then x < C)
                self.model.addConstr(
                    self.variables["x"][z] <= (self.existing_0_12[z] - 1) * (1 - self.variables["big_expansion"][z]) + 
                    self.expand_upper_bound[z] * self.variables["big_expansion"][z],
                    name=f"big_expansion_2_{z}"
                )

    def _set_objective(self):
        """设置目标函数：最小化总成本 (Set objective function: minimize total cost)"""
        # 1. 建设成本 (Construction cost)
        build_cost = quicksum(
            self.cost_build[t] * self.variables["y"][z, t] 
            for z in self.ZIPS for t in self.types
        )
        
        # 2. 设备成本（0-5岁专用） (Equipment cost (dedicated to 0-5 years old))
        equip_cost = quicksum(
            100 * self.variables["z05"][z, t] 
            for z in self.ZIPS for t in self.types
        )
        
        # 3. 扩容成本（基础扩容成本 + 大规模扩容附加成本） (Expansion cost (base expansion cost + large expansion additional cost))
        expand_cost = quicksum(
            200 * self.variables["x"][z] for z in self.ZIPS
        ) + quicksum(
            (20000 + 200 * self.existing_0_12[z]) * self.variables["big_expansion"][z]
            for z in self.ZIPS
        )
        
        # 总成本最小化 (Minimize total cost)
        self.model.setObjective(build_cost + equip_cost + expand_cost, GRB.MINIMIZE)

    def solve(self):
        """求解模型 (Solve model)"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.optimize()
        
        # 检查求解状态 (Check solution status)
        if self.model.status != GRB.OPTIMAL:
            print(f"Model not solved optimally. Status: {self.model.status}")
            return None
        
        return self._extract_results()

    def _extract_results(self):
        """提取结果 (Extract results)"""
        results = []
        for z in self.ZIPS:
            results.append({
                "zip": z,
                "expand": self.variables["x"][z].X,
                "big_expansion": self.variables["big_expansion"][z].X,
                "small": self.variables["y"][z, "small"].X,
                "medium": self.variables["y"][z, "medium"].X,
                "large": self.variables["y"][z, "large"].X,
                "slots05": sum(self.variables["z05"][z, t].X for t in self.types),
                "total_new_capacity": (
                    self.variables["y"][z, "small"].X * 100 +
                    self.variables["y"][z, "medium"].X * 200 +
                    self.variables["y"][z, "large"].X * 400
                )
            })
        
        return pd.DataFrame(results)

    def save_results(self, filename="scenario1_gurobi_solution.csv"):
        """保存结果到CSV文件 (Save results to CSV file)"""
        if "results" not in self.__dict__ or self.results is None:
            self.results = self.solve()
        
        if self.results is not None:
            self.results.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            print(f"Total cost: ${self.model.objVal:,.2f}")
        else:
            print("No results to save.")


# 主程序 (Main program)
if __name__ == "__main__":
    # 创建模型实例 (Create model instance)
    eliminator = ChildCareDesertEliminator()
    
    # 加载并预处理数据 (Load and preprocess data)
    eliminator.load_and_preprocess_data()
    
    # 构建模型 (Build model)
    eliminator.build_model()
    
    # 求解并保存结果 (Solve and save results)
    results = eliminator.solve()
    
    if results is not None:
        print(results)
        eliminator.save_results()
    
    # 打印一些关键信息 (Print some key information)
    print("\nKey Model Information:")
    print(f"Number of ZIP codes: {len(eliminator.ZIPS)}")
    print(f"Total existing capacity (0-12): {sum(eliminator.existing_0_12.values()):,.0f}")
    print(f"Total required capacity (0-12): {sum(eliminator.req_012.values()):,.0f}")
    print(f"Total existing capacity (0-5): {sum(eliminator.existing_0_5.values()):,.0f}")
    print(f"Total required capacity (0-5): {sum(eliminator.req_05.values()):,.0f}")