import pandas as pd
import numpy as np
import math

def load_and_process(
    data_dir="",
    output_file="../Data/Result Data/processed_data.csv",
    distance_file="../Data/Result Data/location_distances.csv",
    facility_locations_file="../Data/Result Data/existing_facilities_locations.csv",
    too_close_file="../Data/Result Data/too_close_positions.csv",
    facility_data_file="../Data/Result Data/facility_data.csv",
):
    """
    Load and process all data, save to CSV files
    :return: Processed data DataFrame
    """
    df_zip, df_potential, facility_locations, facility_data, facility_zip_map = _load_data(data_dir)

    (
        processed_data,
        location_distances,
        too_close_positions,
        df_potential_processed,
        df_zip_with_reqs,
    ) = _preprocess_data(df_zip, df_potential, facility_locations)

    _save_processed_data(
        processed_data=processed_data,
        location_distances=location_distances,
        facility_locations=facility_locations,
        too_close_positions=too_close_positions,
        facility_data=facility_data,
        facility_zip_map=facility_zip_map,
        output_file=output_file,
        distance_file=distance_file,
        facility_locations_file=facility_locations_file,
        too_close_file=too_close_file,
        facility_data_file=facility_data_file,
    )
    return processed_data


def _load_data(data_dir="../Data/Raw data/"):
    """Load all raw data files and perform base processing"""
    df_fac = pd.read_csv(f"{data_dir}child_care_regulated.csv")
    df_fac["facility_id"] = range(len(df_fac))

    df_inc = pd.read_csv(f"{data_dir}avg_individual_income.csv")
    df_pop = pd.read_csv(f"{data_dir}population.csv")
    df_emp = pd.read_csv(f"{data_dir}employment_rate.csv")
    df_potential = pd.read_csv(f"{data_dir}potential_locations.csv")

    _process_zip_codes(df_fac, df_inc, df_pop, df_emp, df_potential)

    _calculate_capacities(df_fac)
    _calculate_population(df_pop)

    df_zip = _merge_data(df_fac, df_inc, df_pop, df_emp)

    df_zip = df_zip.dropna(subset=["existing_capacity_0_12", "pop_0_12", "pop_0_5"])

    facility_locations = _extract_facility_locations(df_fac)
    facility_data = df_fac.copy()
    facility_zip_map = df_fac[["facility_id", "zip_code"]].copy()

    return df_zip, df_potential, facility_locations, facility_data, facility_zip_map


def _extract_facility_locations(df_fac):
    """Extract location information of existing facilities"""
    df_fac = df_fac.copy()
    df_fac["zip_code"] = df_fac["zip_code"].astype(int)
    df_fac = df_fac.dropna(subset=["latitude", "longitude"])
    facility_locations = df_fac[["facility_id", "zip_code", "latitude", "longitude"]].copy()
    return facility_locations


def _process_zip_codes(*dfs):
    """Standardize ZIP code format"""
    for df in dfs:
        if df is None:
            continue
        if "zip_code" in df.columns:
            df.loc[df["zip_code"] >= 100000, "zip_code"] = df["zip_code"] // 10000
        elif "ZIP code" in df.columns:
            df.loc[df["ZIP code"] >= 100000, "ZIP code"] = df["ZIP code"] // 10000
        elif "zipcode" in df.columns:
            df.loc[df["zipcode"] >= 100000, "zipcode"] = df["zipcode"] // 10000


def _calculate_capacities(df_fac):
    """Calculate childcare facility capacities"""
    df_fac["existing_capacity_0_12"] = df_fac["total_capacity"]
    df_fac["existing_capacity_0_5"] = (
        df_fac[["infant_capacity", "toddler_capacity", "preschool_capacity"]].sum(axis=1)
        + (5 / 12) * df_fac["children_capacity"]
    )
    df_fac["initial_capacity_0_12"] = df_fac["existing_capacity_0_12"]
    df_fac["initial_capacity_0_5"] = df_fac["existing_capacity_0_5"]


def _calculate_population(df_pop):
    """Calculate population statistics"""
    df_pop["pop_0_5"] = df_pop["-5"]
    df_pop["10-12"] = df_pop["10-14"] * (3 / 5)
    df_pop["pop_0_12"] = df_pop[["-5", "5-9", "10-12"]].sum(axis=1)


def _merge_data(df_fac, df_inc, df_pop, df_emp):
    """Merge all datasets"""
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
        .merge(pop_by_zip, on="zip_code", how="left")
        .merge(inc_by_zip, on="zip_code", how="left")
        .merge(emp_by_zip, on="zip_code", how="left")
    )
    return df_zip


def _preprocess_data(df_zip, df_potential, facility_locations):
    """Preprocess data, including calculating demand and processing potential locations"""
    df_zip = _calculate_requirements(df_zip)
    df_potential_processed, location_distances = _process_potential_locations(df_potential)
    too_close_positions = _find_too_close_positions(df_potential_processed, facility_locations)
    processed_data = _format_processed_data(df_zip, df_potential_processed)
    return processed_data, location_distances, too_close_positions, df_potential_processed, df_zip


def _calculate_requirements(df_zip):
    """Calculate minimum requirements"""
    df_zip = df_zip.copy()
    df_zip["high_demand"] = (df_zip["average income"] <= 60000) | (df_zip["employment rate"] >= 0.6)
    df_zip["min_required_0_12"] = df_zip.apply(
        lambda r: 0.5 * r["pop_0_12"] if r["high_demand"] else (1 / 3) * r["pop_0_12"],
        axis=1,
    )
    df_zip["min_required_0_5"] = (2 / 3) * df_zip["pop_0_5"]
    return df_zip


def _process_potential_locations(df_potential):
    """Process potential location data, calculate distances between locations"""
    df_potential = df_potential.drop_duplicates(subset=["zipcode", "latitude", "longitude"]).copy()
    df_potential["location_id"] = df_potential.groupby("zipcode").cumcount()
    location_distances = []
    zip_groups = df_potential.groupby("zipcode")
    for zip_code, group in zip_groups:
        if len(group) > 1:
            coords = group[["latitude", "longitude"]].values
            distances = _haversine_distance_matrix(coords)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    d = distances[i, j]
                    location_distances.append(
                        {
                            "zip_code": zip_code,
                            "loc1_id": group.iloc[i]["location_id"],
                            "loc2_id": group.iloc[j]["location_id"],
                            "distance": d,
                            "too_close": d < 0.06,
                        }
                    )
    location_distances = pd.DataFrame(location_distances)
    return df_potential, location_distances


def _haversine_distance_matrix(coords):
    """
    Calculate distances between all point pairs in coordinate matrix (miles)
    :param coords: 2D array, each row is [latitude, longitude]
    :return: Distance matrix
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    R = 3959
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist = R * c
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def _find_too_close_positions(df_potential, facility_locations):
    """Calculate which potential positions are too close to existing facilities"""
    too_close_positions = []
    zip_groups = df_potential.groupby("zipcode")
    for zip_code, group in zip_groups:
        facilities = facility_locations[facility_locations["zip_code"] == zip_code]
        if facilities.empty:
            continue

        coords = group[["latitude", "longitude"]].values
        facility_coords = facilities[["latitude", "longitude"]].values

        for i in range(len(group)):
            for j in range(len(facilities)):
                dist = _haversine_distance(
                    (coords[i][0], coords[i][1]),
                    (facility_coords[j][0], facility_coords[j][1]),
                )
                if dist < 0.06:
                    too_close_positions.append(
                        {
                            "zip_code": zip_code,
                            "location_id": group.iloc[i]["location_id"],
                            "facility_id": facilities.iloc[j]["facility_id"],
                            "too_close_to_facility": True,
                        }
                    )
    return pd.DataFrame(too_close_positions)


def _haversine_distance(coord1, coord2):
    """
    Calculate Haversine distance between two points (miles)
    :param coord1: (latitude, longitude)
    :param coord2: (latitude, longitude)
    :return: Distance (miles)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = R * c
    return dist


def _format_processed_data(df_zip, df_potential):
    """Format processed data into a format suitable for optimization"""
    processed = df_zip.copy()
    fac_count = df_zip["zip_code"].map(
        df_zip.groupby("zip_code")["existing_capacity_0_12"].count()
    )
    processed["existing_facilities"] = fac_count
    loc_count = df_potential.groupby("zipcode").size().reset_index(name="potential_locations")
    processed = processed.merge(loc_count, left_on="zip_code", right_on="zipcode", how="left")
    processed["potential_locations"] = processed["potential_locations"].fillna(0).astype(int)
    processed["expand_upper_bound"] = processed["existing_capacity_0_12"] * 0.2
    return processed


def _save_processed_data(
    *,
    processed_data,
    location_distances,
    facility_locations,
    too_close_positions,
    facility_data,
    facility_zip_map,
    output_file,
    distance_file,
    facility_locations_file,
    too_close_file,
    facility_data_file,
):
    """Save processed data to CSV files"""
    if processed_data is not None:
        processed_data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    if location_distances is not None and not location_distances.empty:
        location_distances.to_csv(distance_file, index=False)
        print(f"Location distances saved to {distance_file}")
    if facility_locations is not None and not facility_locations.empty:
        facility_locations.to_csv(facility_locations_file, index=False)
        print(f"Facility locations saved to {facility_locations_file}")
    if too_close_positions is not None and not too_close_positions.empty:
        too_close_positions.to_csv(too_close_file, index=False)
        print(f"Too close positions saved to {too_close_file}")
    if facility_data is not None and not facility_data.empty:
        facility_data.to_csv(facility_data_file, index=False)
        print(f"Facility detailed data saved to {facility_data_file}")
    if facility_zip_map is not None and not facility_zip_map.empty:
        facility_zip_map.to_csv("facility_zip_map.csv", index=False)
        print("Facility zip code mapping saved to facility_zip_map.csv")


if __name__ == "__main__":
    print("Starting data preprocessing...")
    processed_data = load_and_process()
    print("Data preprocessing completed.")
