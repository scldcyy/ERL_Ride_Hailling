import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


class HexGridProcessor:
    """
    将地理空间数据(Shapefile)和行程数据(CSV)处理成六边形网格格式，
    并确保网格的连通性（消除孤岛）。
    """

    def __init__(self, shapefile_path, trip_data_path, hex_resolution):
        self.shapefile_path = shapefile_path
        self.trip_data_path = trip_data_path
        self.hex_resolution = hex_resolution
        self.gdf_zones = None
        self.gdf_hex = None
        self.location_centroids = None
        self.adjacency = {}
        print(f"--- Initializing HexGridProcessor (H3 Res: {self.hex_resolution}) ---")

    def _create_location_centroids_map(self):
        """创建区域ID(OBJECTID)到中心点(经度, 纬度)的映射"""
        self.location_centroids = {}
        # 注意：需确保Shapefile包含此列，通常NYC数据为'LocationID'或'OBJECTID'
        id_column = 'OBJECTID'

        self.gdf_zones['centroid'] = self.gdf_zones.geometry.centroid
        for _, row in self.gdf_zones.iterrows():
            loc_id = row[id_column]
            centroid = row['centroid']
            self.location_centroids[loc_id] = (centroid.x, centroid.y)  # (Lng, Lat)

    def _generate_hex_grid(self):
        """读取Shapefile并填充六边形"""
        self.gdf_zones = gpd.read_file(self.shapefile_path)
        if self.gdf_zones.crs != 'EPSG:4326':
            self.gdf_zones = self.gdf_zones.to_crs('EPSG:4326')

        self._create_location_centroids_map()

        # 生成覆盖所有区域的六边形
        unified_polygon = self.gdf_zones.geometry.union_all()
        hex_ids = set()

        polys = list(unified_polygon.geoms) if isinstance(unified_polygon, MultiPolygon) else [unified_polygon]

        for poly in polys:
            # 注意：h3.geo_to_h3shape 在新版可能有所不同，此处沿用旧版逻辑
            h3_poly = h3.geo_to_h3shape(poly)
            cells = h3.polygon_to_cells(h3_poly, self.hex_resolution)
            hex_ids.update(cells)

        self._update_gdf_hex(hex_ids)
        print(f"Generated {len(hex_ids)} initial hexagons")

    def _update_gdf_hex(self, hex_ids):
        """根据hex_id集合更新self.gdf_hex"""
        hex_polygons = []
        for h in hex_ids:
            # h3.cell_to_boundary返回 (lat, lng)，shapely需要 (lng, lat)
            boundary_latlng = h3.cell_to_boundary(h)
            boundary_lnglat = [(lng, lat) for lat, lng in boundary_latlng]
            hex_polygons.append(Polygon(boundary_lnglat))

        self.gdf_hex = gpd.GeoDataFrame(
            data={'hex_id': list(hex_ids)},
            geometry=hex_polygons,
            crs="EPSG:4326"
        )

    def _bridge_disconnected_components(self, max_bridge_distance=4):
        """检测孤岛并建立'桥梁'网格以确保连通性"""
        print(f"Running connectivity check (Max Bridge Dist: {max_bridge_distance})...")
        current_hexes = set(self.gdf_hex['hex_id'])

        while True:
            # 1. 构建邻接图
            G = nx.Graph()
            for h in current_hexes:
                G.add_node(h)
                neighbors = h3.grid_ring(h, 1)
                for n in neighbors:
                    if n in current_hexes:
                        G.add_edge(h, n)

            # 2. 检查连通分量
            components = list(nx.connected_components(G))
            if len(components) <= 1:
                break

            components.sort(key=len, reverse=True)
            main_land = list(components[0])
            other_islands = components[1:]

            bridges_built = 0

            # 3. 贪心策略连接孤岛
            for island in other_islands:
                island_list = list(island)
                local_min_dist = float('inf')
                local_best_pair = None

                # 寻找最近点对 (简化版，未做空间索引优化)
                for h_island in island_list:
                    for h_main in main_land:
                        try:
                            d = h3.grid_distance(h_island, h_main)
                            if d < local_min_dist:
                                local_min_dist = d
                                local_best_pair = (h_island, h_main)
                                if d <= 2: break
                        except:
                            continue
                    if local_min_dist <= 2: break

                if local_best_pair and local_min_dist <= max_bridge_distance:
                    start, end = local_best_pair
                    try:
                        bridge_path = h3.grid_path_cells(start, end)
                        current_hexes.update(bridge_path)
                        bridges_built += 1
                    except:
                        pass

            if bridges_built == 0:
                print("Warning: Some islands could not be bridged within max distance.")
                break

        self._update_gdf_hex(current_hexes)
        print(f"Bridging complete. Total hexes: {len(current_hexes)}")

    def _build_adjacency_map(self):
        """建立六边形邻接表"""
        self.adjacency = {}
        valid_hexes = set(self.gdf_hex['hex_id'])
        for hex_id in valid_hexes:
            neighbors = h3.grid_ring(hex_id, 1)
            valid_neighbors = [n for n in neighbors if n in valid_hexes]
            self.adjacency[hex_id] = valid_neighbors

    def _map_trips_to_hex(self):
        """将CSV行程映射到网格"""
        print("Mapping trip data...")
        df_trips = pd.read_csv(self.trip_data_path)

        # 辅助函数：ID -> 经纬度
        def get_coords(loc_id):
            return self.location_centroids.get(loc_id, (None, None))

        # 向量化处理会更快，但这里为了兼容保持 apply
        df_trips['pickup_coords'] = df_trips['PULocationID'].apply(get_coords)
        df_trips['dropoff_coords'] = df_trips['DOLocationID'].apply(get_coords)

        def coords_to_hex(coords):
            lng, lat = coords
            if lng is None or lat is None or pd.isna(lng): return None
            return h3.latlng_to_cell(lat, lng, self.hex_resolution)

        df_trips['pickup_hex_id'] = df_trips['pickup_coords'].apply(coords_to_hex)
        df_trips['dropoff_hex_id'] = df_trips['dropoff_coords'].apply(coords_to_hex)

        df_trips.dropna(subset=['pickup_hex_id', 'dropoff_hex_id'], inplace=True)
        # 清理临时列
        df_trips.drop(columns=['pickup_coords', 'dropoff_coords'], inplace=True)

        return df_trips

    def process(self):
        self._generate_hex_grid()
        self._bridge_disconnected_components()  # 调整距离参数
        self._build_adjacency_map()
        df_trips = self._map_trips_to_hex()
        return df_trips, self.gdf_hex


class PassengerSimulator:
    """ 基于历史数据统计规律的乘客订单生成器 """

    def __init__(self, df_gridded_trips, adjacency, scaling_factor=1.0):
        self.df = df_gridded_trips
        self.adjacency = adjacency
        self.scaling_factor = scaling_factor
        self.demand_model = {}  # (hour, hex) -> lambda
        self.transition_model = {}  # (hour, origin) -> ([destinations], [probs])
        self.trip_props_model = {}  # (origin, dest) -> {dist, duration}

        self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
        self.df['hour'] = self.df['pickup_datetime'].dt.hour
        self._learn_distributions()

    def _learn_distributions(self):
        print("Learning demand distributions...")
        num_days = self.df['pickup_datetime'].dt.date.nunique() or 1

        # 1. 需求分布 (泊松参数 lambda)
        demand_counts = self.df.groupby(['hour', 'pickup_hex_id']).size() / num_days
        self.demand_model = demand_counts.to_dict()

        # 2. 转移概率 (Origin -> Destination)
        transitions = self.df.groupby(['hour', 'pickup_hex_id', 'dropoff_hex_id']).size().reset_index(name='count')
        for (hour, origin), group in transitions.groupby(['hour', 'pickup_hex_id']):
            total = group['count'].sum()
            self.transition_model[(hour, origin)] = (
                group['dropoff_hex_id'].values,
                group['count'].values / total
            )

        # 3. 行程属性 (距离和时间)
        dist_col = 'trip_distance' if 'trip_distance' in self.df.columns else 'trip_miles'
        dur_col = 'trip_duration' if 'trip_duration' in self.df.columns else 'trip_time'
        self.trip_props_model = self.df.groupby(['pickup_hex_id', 'dropoff_hex_id'])[
            [dist_col, dur_col]].mean().to_dict('index')

    def generate_orders(self, time_slot, all_hex_ids):
        all_orders = []
        for hex_id in all_hex_ids:
            lambda_val = self.demand_model.get((time_slot, hex_id), 0)
            if lambda_val <= 0: continue

            # 泊松采样
            num_requests = np.random.poisson(lambda_val * self.scaling_factor)
            if num_requests == 0: continue

            trans_data = self.transition_model.get((time_slot, hex_id))
            if not trans_data: continue

            dests = np.random.choice(trans_data[0], size=num_requests, p=trans_data[1])

            for dest_hex in dests:
                props = self.trip_props_model.get((hex_id, dest_hex), {'trip_distance': 1.0, 'trip_duration': 5})
                # 兼容不同的列名
                dist = props.get('trip_distance', props.get('trip_miles', 1.0))
                dur = props.get('trip_duration', props.get('trip_time', 5.0))

                all_orders.append({
                    'origin_hex': hex_id,
                    'dest_hex': dest_hex,
                    'distance': dist,
                    'duration': dur
                })
        return all_orders

    def real_orders(self):
        """统计真实数据的订单空间分布"""
        return self.df['pickup_hex_id'].value_counts().to_dict()


if __name__ == '__main__':
    # 示例配置
    SHAPEFILE_PATH = 'taxi_zones/taxi_zones.shp'
    TRIP_DATA_PATH = 'fhvhv_jan_01.csv'  # 需确保文件存在
    HEX_RES = 7

    processor = HexGridProcessor(SHAPEFILE_PATH, TRIP_DATA_PATH, HEX_RES)
    df_trips, gdf_hex = processor.process()

    simulator = PassengerSimulator(df_trips, processor.adjacency)

    # 生成测试订单
    orders = simulator.generate_orders(time_slot=10, all_hex_ids=gdf_hex['hex_id'].tolist())
    print(f"Generated {len(orders)} orders for hour 10.")

    # 简单绘图
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_hex.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5)
    processor.gdf_zones.boundary.plot(ax=ax, color='gray', alpha=0.5)
    ax.set_title(f"Processed Hex Grid (Res {HEX_RES})")
    plt.savefig('hex_grid_final.png')
    print("Map saved.")