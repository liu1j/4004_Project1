import pandas as pd
import numpy as np
import math
from gurobipy import Model, GRB, quicksum

class DataLoader:
    """负责加载和预处理数据的类，将结果存入CSV文件
    Class responsible for loading and preprocessing data, saving results to CSV files"""
    
    def __init__(self, data_dir=""):
        """
        初始化DataLoader
        Initialize DataLoader
        
        :param data_dir: 数据文件所在目录
                         Data directory where files are located
        """
        self.data_dir = data_dir
        self.df_zip = None
        self.df_potential = None
        self.processed_data = None
        self.location_distances = None
        self.facility_locations = None  # 存储现有设施位置
                                      # Store existing facility locations
        self.too_close_positions = None  # 存储与现有设施太近的位置
                                       # Store positions too close to existing facilities
    
    def load_and_process(self, output_file="processed_data.csv", 
                         distance_file="location_distances.csv",
                         facility_locations_file="existing_facilities_locations.csv",
                         too_close_file="too_close_positions.csv"):
        """
        加载并处理所有数据，保存到CSV文件
        Load and process all data, save to CSV files
        
        :param output_file: 处理后的数据输出文件名
                            Output file name for processed data
        :param distance_file: 位置距离数据输出文件名
                              Output file name for location distance data
        :param facility_locations_file: 现有设施位置输出文件名
                                        Output file name for existing facility locations
        :param too_close_file: 太近位置输出文件名
                               Output file name for too close positions
        :return: 处理后的数据DataFrame
                 Processed data DataFrame
        """
        # 加载原始数据
        # Load raw data
        self._load_data()
        
        # 预处理数据
        # Preprocess data
        self._preprocess_data()
        
        # 保存处理后的数据
        # Save processed data
        self._save_processed_data(output_file, distance_file, facility_locations_file, too_close_file)
        
        return self.processed_data
    
    def _load_data(self):
        """加载所有原始数据文件
        Load all raw data files"""
        # 加载托儿机构数据
        # Load childcare facility data
        df_fac = pd.read_csv(f"{self.data_dir}child_care_regulated.csv")
        
        # 加载收入数据
        # Load income data
        df_inc = pd.read_csv(f"{self.data_dir}avg_individual_income.csv")
        
        # 加载人口数据
        # Load population data
        df_pop = pd.read_csv(f"{self.data_dir}population.csv")
        
        # 加载就业率数据
        # Load employment rate data
        df_emp = pd.read_csv(f"{self.data_dir}employment_rate.csv")
        
        # 加载潜在位置数据
        # Load potential location data
        self.df_potential = pd.read_csv(f"{self.data_dir}potential_locations.csv")
        
        # 处理ZIP码格式
        # Process ZIP code format
        self._process_zip_codes(df_fac, df_inc, df_pop, df_emp, self.df_potential)
        
        # 计算容量
        # Calculate capacities
        self._calculate_capacities(df_fac)
        
        # 计算人口
        # Calculate population
        self._calculate_population(df_pop)
        
        # 合并数据
        # Merge data
        self.df_zip = self._merge_data(df_fac, df_inc, df_pop, df_emp)
        
        # 删除关键字段缺失的行
        # Remove rows with missing key fields
        self.df_zip = self.df_zip.dropna(subset=["existing_capacity_0_12", "pop_0_12", "pop_0_5"])
        
        # 保存现有设施位置
        # Save existing facility locations
        self.facility_locations = self._extract_facility_locations(df_fac)
    
    def _extract_facility_locations(self, df_fac):
        """提取现有设施的位置信息
        Extract location information of existing facilities"""
        # 确保zip_code是整数类型
        # Ensure zip_code is integer type
        df_fac["zip_code"] = df_fac["zip_code"].astype(int)
        
        # 只保留有经纬度数据的行
        # Keep only rows with latitude and longitude data
        df_fac = df_fac.dropna(subset=["latitude", "longitude"])
        
        # 选择需要的列
        # Select required columns
        facility_locations = df_fac[["zip_code", "latitude", "longitude"]].copy()
        
        return facility_locations
    
    def _process_zip_codes(self, *dfs):
        """标准化ZIP码格式
        Standardize ZIP code format"""
        for df in dfs:
            if df is None:
                continue
                
            if "zip_code" in df.columns:
                df.loc[df["zip_code"] >= 100000, "zip_code"] = df["zip_code"] // 10000
            elif "ZIP code" in df.columns:
                df.loc[df["ZIP code"] >= 100000, "ZIP code"] = df["ZIP code"] // 10000
            elif "zipcode" in df.columns:
                df.loc[df["zipcode"] >= 100000, "zipcode"] = df["zipcode"] // 10000
    
    def _calculate_capacities(self, df_fac):
        """计算托儿机构容量
        Calculate childcare facility capacities"""
        df_fac["existing_capacity_0_12"] = (
            df_fac[["infant_capacity", "toddler_capacity", "preschool_capacity"]].sum(axis=1)
            + (5/12) * df_fac["children_capacity"]
        )
        df_fac["existing_capacity_0_5"] = df_fac[["infant_capacity", "toddler_capacity"]].sum(axis=1)
    
    def _calculate_population(self, df_pop):
        """计算人口统计数据
        Calculate population statistics"""
        df_pop["pop_0_5"] = df_pop["-5"]
        df_pop["10-12"] = df_pop["10-14"] * (3/5)
        df_pop["pop_0_12"] = df_pop[["-5", "5-9", "10-12"]].sum(axis=1)
    
    def _merge_data(self, df_fac, df_inc, df_pop, df_emp):
        """合并所有数据集
        Merge all datasets"""
        cap_by_zip = (
            df_fac.groupby("zip_code")[["existing_capacity_0_12", "existing_capacity_0_5"]]
            .sum()
            .reset_index()
        )
        
        pop_by_zip = df_pop[["zipcode", "pop_0_5", "pop_0_12"]].rename(columns={"zipcode": "zip_code"})
        inc_by_zip = df_inc.rename(columns={"ZIP code": "zip_code"})
        emp_by_zip = df_emp.rename(columns={"zipcode": "zip_code"})
        
        df_zip = (
            cap_by_zip
            .merge(pop_by_zip, on="zip_code", how="outer")
            .merge(inc_by_zip, on="zip_code", how="outer")
            .merge(emp_by_zip, on="zip_code", how="outer")
        )
        
        return df_zip
    
    def _preprocess_data(self):
        """预处理数据，包括计算需求和处理潜在位置
        Preprocess data, including calculating demand and processing potential locations"""
        # 计算需求
        # Calculate demand
        self._calculate_requirements()
        
        # 处理潜在位置数据
        # Process potential location data
        self._process_potential_locations()
        
        # 计算与现有设施太近的位置
        # Calculate positions too close to existing facilities
        self._find_too_close_positions()
        
        # 将处理后的数据整理为便于优化的形式
        # Format processed data for optimization
        self._format_processed_data()
    
    def _calculate_requirements(self):
        """计算最低需求
        Calculate minimum requirements"""
        self.df_zip["high_demand"] = (self.df_zip["average income"] <= 60000) | (self.df_zip["employment rate"] >= 0.6)
        self.df_zip["min_required_0_12"] = self.df_zip.apply(
            lambda r: 0.5 * r["pop_0_12"] if r["high_demand"] else (1/3) * r["pop_0_12"],
            axis=1
        )
        self.df_zip["min_required_0_5"] = (2/3) * self.df_zip["pop_0_5"]
    
    def _process_potential_locations(self):
        """处理潜在位置数据，计算位置之间的距离
        Process potential location data, calculate distances between locations"""
        # 确保没有重复的位置
        # Ensure no duplicate positions
        self.df_potential = self.df_potential.drop_duplicates(subset=["zipcode", "latitude", "longitude"])
        
        # 为每个ZIP码内的位置分配唯一ID
        # Assign unique ID to positions within each ZIP code
        self.df_potential["location_id"] = self.df_potential.groupby("zipcode").cumcount()
        
        # 创建位置距离DataFrame
        # Create location distance DataFrame
        location_distances = []
        
        # 按ZIP码分组处理
        # Process by ZIP code grouping
        zip_groups = self.df_potential.groupby("zipcode")
        
        for zip_code, group in zip_groups:
            if len(group) > 1:
                # 计算Haversine距离矩阵
                # Calculate Haversine distance matrix
                coords = group[["latitude", "longitude"]].values
                distances = self._haversine_distance_matrix(coords)
                
                # 转换为长格式
                # Convert to long format
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        location_distances.append({
                            "zip_code": zip_code,
                            "loc1_id": group.iloc[i]["location_id"],
                            "loc2_id": group.iloc[j]["location_id"],
                            "distance": distances[i, j],
                            "too_close": distances[i, j] < 0.06
                        })
        
        self.location_distances = pd.DataFrame(location_distances)
    
    def _haversine_distance_matrix(self, coords):
        """
        计算坐标矩阵中所有点对之间的距离（英里）
        Calculate distances between all point pairs in coordinate matrix (miles)
        
        :param coords: 二维数组，每行是[latitude, longitude]
                       2D array, each row is [latitude, longitude]
        :return: 距离矩阵
                 Distance matrix
        """
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        
        # 地球半径（英里）
        # Earth radius (miles)
        R = 3959
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                
                # 转换为弧度
                # Convert to radians
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                
                # Haversine公式
                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                dist = R * c
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def _find_too_close_positions(self):
        """计算哪些潜在位置与现有设施太近
        Calculate which potential positions are too close to existing facilities"""
        too_close_positions = []
        
        # 按ZIP码分组处理
        # Process by ZIP code grouping
        zip_groups = self.df_potential.groupby("zipcode")
        
        for zip_code, group in zip_groups:
            # 获取该ZIP码的现有设施位置
            # Get existing facility locations for this ZIP code
            facilities = self.facility_locations[self.facility_locations["zip_code"] == zip_code]
            
            if facilities.empty:
                continue
                
            # 获取该ZIP码的潜在位置
            # Get potential locations for this ZIP code
            coords = group[["latitude", "longitude"]].values
            facility_coords = facilities[["latitude", "longitude"]].values
            
            # 对每个潜在位置，检查是否与任何现有设施太近
            # For each potential location, check if it's too close to any existing facility
            for i in range(len(group)):
                for j in range(len(facilities)):
                    dist = self._haversine_distance(
                        (coords[i][0], coords[i][1]),
                        (facility_coords[j][0], facility_coords[j][1])
                    )
                    
                    if dist < 0.06:  # 小于0.06英里
                                     # Less than 0.06 miles
                        too_close_positions.append({
                            "zip_code": zip_code,
                            "location_id": group.iloc[i]["location_id"],
                            "too_close_to_facility": True
                        })
        
        self.too_close_positions = pd.DataFrame(too_close_positions)
    
    def _haversine_distance(self, coord1, coord2):
        """
        计算两点之间的Haversine距离（英里）
        Calculate Haversine distance between two points (miles)
        
        :param coord1: (latitude, longitude)
        :param coord2: (latitude, longitude)
        :return: 距离（英里）
                 Distance (miles)
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # 地球半径（英里）
        # Earth radius (miles)
        R = 3959
        
        # 转换为弧度
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine公式
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        dist = R * c
        
        return dist
    
    def _format_processed_data(self):
        """将处理好的数据整理成适合优化的格式
        Format processed data into a format suitable for optimization"""
        # 创建包含所有必要信息的DataFrame
        # Create DataFrame containing all necessary information
        processed = self.df_zip.copy()
        
        # 添加现有设施数量
        # Add existing facility count
        fac_count = self.df_zip["zip_code"].map(
            self.df_zip.groupby("zip_code")["existing_capacity_0_12"].count()
        )
        processed["existing_facilities"] = fac_count
        
        # 添加潜在位置信息
        # Add potential location information
        loc_count = self.df_potential.groupby("zipcode").size().reset_index(name="potential_locations")
        processed = processed.merge(loc_count, left_on="zip_code", right_on="zipcode", how="left")
        processed["potential_locations"] = processed["potential_locations"].fillna(0).astype(int)
        
        # 为每个ZIP码添加扩容上限（20%）
        # Add expansion upper bound for each ZIP code (20%)
        processed["expand_upper_bound"] = processed["existing_capacity_0_12"] * 0.2
        
        self.processed_data = processed
    
    def _save_processed_data(self, output_file, distance_file, facility_locations_file, too_close_file):
        """保存处理好的数据到CSV文件
        Save processed data to CSV files"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_file, index=False)
            print(f"Processed data saved to {output_file}")
        
        if self.location_distances is not None and not self.location_distances.empty:
            self.location_distances.to_csv(distance_file, index=False)
            print(f"Location distances saved to {distance_file}")
        
        if self.facility_locations is not None and not self.facility_locations.empty:
            self.facility_locations.to_csv(facility_locations_file, index=False)
            print(f"Facility locations saved to {facility_locations_file}")
        
        if self.too_close_positions is not None and not self.too_close_positions.empty:
            self.too_close_positions.to_csv(too_close_file, index=False)
            print(f"Too close positions saved to {too_close_file}")


# 主程序
# Main program
if __name__ == "__main__":
    print("Starting data preprocessing...")
    # 开始数据预处理
    data_loader = DataLoader(data_dir="./")
    processed_data = data_loader.load_and_process()
    print("Data preprocessing completed.")
    # 数据预处理完成