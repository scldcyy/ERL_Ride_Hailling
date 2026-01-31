import os

import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import geopandas as gpd

class DataSetProcesser:
    def __init__(self):
        self.hourly_counts = None
        self.superzone_counts = None
        self.df_zones_info = None
        self.df_trips = None
        self.valid_gdf_zones= None
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.trip_data_file = os.path.join(cur_dir,'fhvhv_jan_01.parquet')
        self.zone_lookup_file = os.path.join(cur_dir,'taxi+_zone_lookup.csv')
        self.shapefile_dir = os.path.join(cur_dir,'taxi_zones')
        self.shapefile_path = os.path.join(self.shapefile_dir, 'taxi_zones.shp')
        self.n_zones=40
        self._dataLoading()

    # Convert parquet to csv
    @staticmethod
    def process2csv(name='fhvhv_tripdata_2024-01.parquet'):
        trips = pq.read_table(name)
        trips = trips.to_pandas()
        trips.to_csv(name.replace('.parquet', '.csv'), index=False)

    # 将1月数据集裁剪为1天
    @staticmethod
    def cutDataSet(full_dataset_file = 'fhvhv_tripdata_2024-01.parquet', filtered_dataset_file = 'fhvhv_jan_01.parquet'):
        # Define file paths

        print("--- Task: Filtering dataset to a single day (Jan 1, 2024) ---")

        try:
            # --- Step 1: Load the full dataset ---
            # To handle very large files efficiently, we can process it in chunks.
            # However, for a one-time filtering, loading it all at once is simpler if memory allows.
            print(f"Loading full dataset from '{full_dataset_file}'...")
            df_full = pd.read_parquet(full_dataset_file)
            print(f"Full dataset loaded with {len(df_full)} records.")

            # --- Step 2: Convert 'pickup_datetime' to datetime objects ---
            # We will coerce errors, turning any unparseable dates into NaT (Not a Time)
            df_full['pickup_datetime'] = pd.to_datetime(df_full['pickup_datetime'], errors='coerce')

            # Drop rows where date conversion failed
            df_full.dropna(subset=['pickup_datetime'], inplace=True)

            # --- Step 3: Perform the date filtering ---
            # Create a boolean mask for the target date
            target_date = pd.to_datetime('2024-01-15').date()
            mask = df_full['pickup_datetime'].dt.date == target_date

            # Apply the mask to filter the DataFrame
            df_jan_01 = df_full[mask].copy()  # Use .copy() to avoid SettingWithCopyWarning

            print(f"Filtered dataset to {len(df_jan_01)} records for the date {target_date}.")

            # --- Step 4: Save the new, smaller dataset to a new CSV file ---
            df_jan_01.to_parquet(filtered_dataset_file, index=False)
            print(f"Filtered data saved to '{filtered_dataset_file}'.")
            print("You can now use this smaller file for faster development and simulation.")

        except FileNotFoundError:
            print(f"Error: The file '{full_dataset_file}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # --- 导入数据，预处理日期 ---
    def _dataLoading(self):
        try:
            self.df_trips = pd.read_parquet(self.trip_data_file)
            self.df_zones_info = pd.read_csv(self.zone_lookup_file)
            self.df_trips['pickup_datetime'] = pd.to_datetime(self.df_trips['pickup_datetime'])
            self.df_trips['hour'] = self.df_trips['pickup_datetime'].dt.hour
            self._geospatialClustering()
            self.hourly_counts = self.df_trips['hour'].value_counts().sort_index()
            self.superzone_counts = self.df_trips['PU_SuperZone'].value_counts().sort_index()
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure dataset.csv and taxi+_zone_lookup.csv are in the same directory.")
            exit()

    # 划分区域
    def _geospatialClustering(self):
        # --- 2. Geospatial Clustering into 40 Sub-regions (REVISED AND CORRECTED) ---
        print("--- Task 2: Performing Geographically Accurate Clustering ---")
        gdf_zones = gpd.read_file(self.shapefile_path)

        print("Columns in the shapefile:", gdf_zones.columns)

        # Project from geographic (lat/lon) to a projected (planar) CRS for accurate calculations.
        # EPSG:2263 is a common choice for New York.
        gdf_zones_proj = gdf_zones.to_crs("EPSG:2263")
        gdf_zones['centroid'] = gdf_zones_proj['geometry'].centroid.to_crs(gdf_zones.crs)  # Project centroid back for plotting

        gdf_zones['longitude'] = gdf_zones['centroid'].x
        gdf_zones['latitude'] = gdf_zones['centroid'].y

        # Filter out invalid zones
        # The column name for Location ID in the shapefile is 'LocationID' (case-sensitive)
        self.valid_gdf_zones = gdf_zones[gdf_zones['LocationID'] <= 263].copy()
        coordinates = self.valid_gdf_zones[['latitude', 'longitude']].values

        kmeans = KMeans(n_clusters=self.n_zones, random_state=42, n_init=10)
        self.valid_gdf_zones['SuperZone'] = kmeans.fit_predict(coordinates)

        # Create the mapping from the correct column 'LocationID'
        zone_to_superzone_map = self.valid_gdf_zones.set_index('LocationID')['SuperZone'].to_dict()

        self.df_trips['PU_SuperZone'] = self.df_trips['PULocationID'].map(zone_to_superzone_map)
        self.df_trips['DO_SuperZone'] = self.df_trips['DOLocationID'].map(zone_to_superzone_map)

        self.df_trips.dropna(subset=['PU_SuperZone', 'DO_SuperZone'], inplace=True)
        self.df_trips['PU_SuperZone'] = self.df_trips['PU_SuperZone'].astype(int)
        self.df_trips['DO_SuperZone'] = self.df_trips['DO_SuperZone'].astype(int)

        print("Geographic clustering complete.")

    # 绘制直方图，统计时间段
    def timePeriodStat(self, save_path='stat/time_period_stat.png'):
        print("--- Task 1: Analyzing Order Volume by Hour ---")
        plt.figure(figsize=(12, 7))
        sns.barplot(x=self.hourly_counts.index, y=self.hourly_counts.values, palette="viridis")
        plt.title('NYC Ride-Hailing Order Volume by Hour (Jan 1 2024)', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_path)
        print("Hourly order volume analysis complete.\n")

    # 绘制区域分布
    def zoneLookup(self, save_path='stat/zone_lookup.png'):
        # --- Visualization of the new clusters ---
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        self.valid_gdf_zones.plot(column='SuperZone', ax=ax, legend=True, cmap='tab20', categorical=True)
        plt.title(f'NYC Taxi Zones Clustered into {self.n_zones} Super Zones', fontsize=16)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(save_path)
        print("Cluster visualization displayed.\n")

    # 绘制区域订单量统计
    def subRegionStat(self, save_path='stat/sub_region_stat.png'):
        # 绘制出发地分块结果
        plt.figure(figsize=(15, 8))
        sns.barplot(x=self.superzone_counts.index, y=self.superzone_counts.values, palette="plasma")
        plt.title('Total Order Volume by Geographically Clustered Super Zone', fontsize=16)
        plt.xlabel('Super Zone ID (0-39)', fontsize=12)
        plt.ylabel('Total Number of Orders', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_path)
        print("Super Zone order volume analysis complete.\n")


if __name__ == '__main__':
    # 数据集裁剪
    # DataSetProcesser.cutDataSet()

    # # --绘制统计图--#
    # dataSetProcesser = DataSetProcesser()
    # dataSetProcesser.timePeriodStat()
    # dataSetProcesser.zoneLookup()
    # dataSetProcesser.subRegionStat()
    pass

