import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os

# --- 配置 ---
# 确保 shapefile 文件夹 (taxi_zones) 和 CSV 文件在同一个目录
SHAPEFILE_PATH = 'taxi_zones/taxi_zones.shp'
TRIP_DATA_PATH = 'fhvhv_jan_01.csv'
HEX_RESOLUTION = 7


class HexGridProcessor:
    """ 一个将地理空间数据和行程数据处理成六边形网格格式的工具。 """

    def __init__(self, shapefile_path, trip_data_path, hex_resolution):
        """ 初始化处理器。 """
        self.shapefile_path = shapefile_path
        self.trip_data_path = trip_data_path
        self.hex_resolution = hex_resolution
        self.gdf_zones = None
        self.location_centroids = None
        print(f"--- Initializing HexGridProcessor with H3 Resolution: {self.hex_resolution} ---")

    def _generate_hex_grid(self):
        """ 生成覆盖Shapefile区域的六边形网格。 """
        try:
            self.gdf_zones = gpd.read_file(self.shapefile_path)
        except Exception as e:
            print(f"错误：读取 shapefile 文件 '{self.shapefile_path}'失败: {e}")
            print("请确保上传了完整的 shapefile 文件，包括 .shp, .shx, 和 .dbf 文件。")
            return False

        if self.gdf_zones.crs != 'EPSG:4326':
            self.gdf_zones = self.gdf_zones.to_crs('EPSG:4326')
            print("已将 Shapefile 文件的坐标系重投影到 EPSG:4326。")

        self._create_location_centroids_map()

        unified_polygon = self.gdf_zones.geometry.unary_union

        hex_ids = set()
        # 将统一后的多边形转换为 H3 可接受的 GeoJSON 格式
        geojson_polygon = gpd.GeoSeries([unified_polygon]).__geo_interface__

        # polygon_to_cells 期望一个GeoJSON-like的字典结构
        cells = h3.polygon_to_cells(geojson_polygon['features'][0]['geometry'], self.hex_resolution)
        hex_ids.update(cells)

        print(f"生成了 {len(hex_ids)} 个六边形。")

        hex_polygons = [Polygon(h3.cell_to_boundary(h, geo_json=True)) for h in hex_ids]

        self.gdf_hex = gpd.GeoDataFrame(data={'hex_id': list(hex_ids)}, geometry=hex_polygons, crs="EPSG:4326")
        self.gdf_hex = self.gdf_hex[self.gdf_hex.geometry.is_valid & ~self.gdf_hex.geometry.is_empty]
        print(f"清理后剩余 {len(self.gdf_hex)} 个有效六边形。")
        return True

    def _create_location_centroids_map(self):
        """创建区域ID到其中心点经纬度的映射。"""
        print("   正在创建 Location ID 到中心点的映射...")
        id_column = 'LocationID'
        if id_column not in self.gdf_zones.columns:
            print(f"错误: Shapefile 中未找到 '{id_column}' 列。")
            return

        self.gdf_zones['centroid'] = self.gdf_zones.geometry.centroid
        self.location_centroids = {row[id_column]: (row['centroid'].x, row['centroid'].y) for _, row in
                                   self.gdf_zones.iterrows()}
        print(f"✅ 已为 {len(self.location_centroids)} 个区域创建中心点映射。")

    def _map_trips_to_hex(self):
        """ 将CSV中的行程数据映射到六边形网格上。 """
        print("正在从 CSV 映射行程数据到六边形网格...")
        try:
            df_trips = pd.read_csv(self.trip_data_path)
        except FileNotFoundError:
            print(f"错误: 未找到行程数据文件 '{self.trip_data_path}'。")
            print("请确保 'fhvhv_jan_01.csv' 文件存在。")
            return None

        required_cols = ['PULocationID', 'DOLocationID']
        if not all(col in df_trips.columns for col in required_cols):
            print(f"错误: CSV文件必须包含 {required_cols} 列。")
            return None

        df_trips['PULocationID'] = pd.to_numeric(df_trips['PULocationID'], errors='coerce')
        df_trips.dropna(subset=['PULocationID'], inplace=True)
        df_trips['PULocationID'] = df_trips['PULocationID'].astype(int)

        df_trips['pickup_centroid'] = df_trips['PULocationID'].map(self.location_centroids)
        df_trips.dropna(subset=['pickup_centroid'], inplace=True)

        df_trips['pickup_longitude'] = df_trips['pickup_centroid'].apply(lambda x: x[0])
        df_trips['pickup_latitude'] = df_trips['pickup_centroid'].apply(lambda x: x[1])

        df_trips['pickup_hex_id'] = df_trips.apply(
            lambda row: h3.latlng_to_cell(row['pickup_latitude'], row['pickup_longitude'], self.hex_resolution),
            axis=1
        )

        initial_count = len(df_trips)
        df_trips.dropna(subset=['pickup_hex_id'], inplace=True)
        mapped_count = len(df_trips)

        print(f"成功将 {mapped_count} / {initial_count} 条行程映射到六边形网格。")
        return df_trips

    def process(self):
        if not self._generate_hex_grid():
            return None, None
        df_gridded_trips = self._map_trips_to_hex()
        if df_gridded_trips is None:
            return None, None
        print("--- HexGridProcessor 完成。 ---")
        return df_gridded_trips, self.gdf_hex


# --- 主脚本 ---
if __name__ == '__main__':
    try:
        # 1. 处理数据
        processor = HexGridProcessor(SHAPEFILE_PATH, TRIP_DATA_PATH, HEX_RESOLUTION)
        df_gridded_trips, gdf_hex = processor.process()

        if df_gridded_trips is not None and gdf_hex is not None:
            # 2. 聚合实际订单数量
            print("\n正在聚合每个六边形的实际订单数量...")
            actual_order_counts = df_gridded_trips.groupby('pickup_hex_id').size().reset_index(name='order_count')
            print(f"在 {len(actual_order_counts)} 个独立的六边形中发现了订单。")

            # 3. 合并订单数量与六边形地理信息
            print("正在合并订单数量与六边形地理信息...")
            gdf_hex_with_counts = gdf_hex.merge(actual_order_counts, left_on='hex_id', right_on='pickup_hex_id',
                                                how='left')
            gdf_hex_with_counts['order_count'] = gdf_hex_with_counts['order_count'].fillna(0)

            # 4. 生成热力图
            print("正在生成热力图...")
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))

            gdf_hex_with_counts.plot(column='order_count',
                                     ax=ax,
                                     legend=True,
                                     cmap='viridis',
                                     legend_kwds={'label': "实际订单数量", 'orientation': "horizontal"})

            if processor.gdf_zones is not None:
                processor.gdf_zones.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.5)

            # 5. 自定义并保存图像
            ax.set_title(f'网格实际订单数量热力图 (H3 分辨率: {HEX_RESOLUTION})', fontsize=16)
            ax.set_xlabel("经度", fontsize=12)
            ax.set_ylabel("纬度", fontsize=12)
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True, linestyle='--', alpha=0.3)

            output_filename = 'grid_order_heatmap.png'
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"\n✅ 热力图已成功保存为 '{output_filename}'")
            plt.show()

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        print("请检查上面的控制台输出以获取有关文件丢失或数据问题的具体错误。")

