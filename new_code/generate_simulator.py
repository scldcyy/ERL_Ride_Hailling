import os
import pickle
import random

import holidays
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

from basic_config import CONFIG


class HexGridProcessor:
    """
    将地理空间数据(Shapefile)和行程数据(CSV)处理成六边形网格格式，
    """

    def __init__(self, shapefile_path, trip_data_path, hex_resolution):
        self.shapefile_path = shapefile_path
        self.trip_data_path = trip_data_path
        self.hex_resolution = hex_resolution
        self.avg_v = 0
        self.gdf_zones = None
        self.gdf_hex = None
        self.location_centroids = None
        self.adjacency = {}
        print(f"--- Initializing HexGridProcessor (H3 Res: {self.hex_resolution}) ---")

    def _create_location_centroids_map(self):
        """创建区域ID(OBJECTID)到中心点(经度, 纬度)的映射"""
        self.location_centroids = {}
        self.gdf_zones['centroid'] = self.gdf_zones.geometry.centroid
        for _, row in self.gdf_zones.iterrows():
            loc_id = row['OBJECTID']
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
            h3_poly = h3.geo_to_h3shape(poly)
            cells = h3.polygon_to_cells_experimental(h3_poly, self.hex_resolution,'bbox_overlap')
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

    def _build_adjacency_map(self):
        """建立六边形邻接表"""
        self.adjacency = {}
        valid_hexes = set(self.gdf_hex['hex_id'])
        for hex_id in valid_hexes:

            valid_neighbors = {}
            edges = h3.origin_to_directed_edges(hex_id)
            for direction_index, edge in enumerate(edges):
                if edge == 0:
                    continue  # 某些五边形边界情况可能没有6个邻居
                # 2. 从单向边获取目标网格
                neighbor_hex = h3.get_directed_edge_destination(edge)
                if neighbor_hex in valid_hexes:
                    valid_neighbors[direction_index] = neighbor_hex
            self.adjacency[hex_id] = valid_neighbors

    def _map_trips_to_hex(self):
        """将parquet行程映射到网格"""
        print("Mapping trip data...")
        df_trips = pd.read_parquet(self.trip_data_path)

        self.avg_v=1.609344*df_trips['trip_miles'].sum()/df_trips['trip_time'].sum()  #0.007644803636864813km/s
        self.avg_t=2.4/self.avg_v

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
        self._build_adjacency_map()
        df_trips = self._map_trips_to_hex()
        return df_trips, self.gdf_hex


class PassengerSimulator:
    """ 基于历史数据统计规律的乘客订单生成器 """

    def __init__(self, df_gridded_trips, adjacency, scaling_factor=1.0):
        self.df = df_gridded_trips
        self.adjacency = adjacency
        self.scaling_factor = scaling_factor
        self.time_step = CONFIG['TIME_STEP_MINUTES']
        self.num_steps = 24 * 60 // self.time_step
        self.demand_model = {}  # (time_step, hex) -> lambda
        self.transition_model = {}  # (time_step, origin) -> ([destinations], [probs])
        self.trip_props_model = {}  # (origin, dest) -> {dist, duration}
        self.all_hex_ids= list(self.adjacency.keys())
        self.fixed_orders = None # 固定生成的订单
        self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
        self.df['pickup_minute_step'] = (self.df['pickup_datetime'].dt.hour* 60 + self.df['pickup_datetime'].dt.minute)//self.time_step
        self._learn_distributions()
        self._init_fixed_orders()

    def _learn_distributions(self):
        print("Learning demand distributions...")
        num_days = self.df['pickup_datetime'].dt.date.nunique() or 1

        # 1. 需求分布 (泊松参数 lambda)
        demand_counts = self.df.groupby(['pickup_minute_step', 'pickup_hex_id']).size() / num_days
        self.demand_model = demand_counts.to_dict()

        # 2. 转移概率 (Origin -> Destination)
        transitions = self.df.groupby(['pickup_minute_step', 'pickup_hex_id', 'dropoff_hex_id']).size().reset_index(name='count')
        for (pickup_minute_step, origin), group in transitions.groupby(['pickup_minute_step', 'pickup_hex_id']):
            total = group['count'].sum()
            self.transition_model[(pickup_minute_step, origin)] = (
                group['dropoff_hex_id'].values,
                group['count'].values / total
            )

        # 3. 行程属性 (距离和时间)
        self.trip_props_model = self.df.groupby(['pickup_hex_id', 'dropoff_hex_id'])[
            ['trip_miles', 'trip_time']].mean().to_dict('index')

    def generate_dynamic_orders(self, time_slot):
        all_orders = []
        for hex_id in self.all_hex_ids:
            # 估计泊松分布的均值
            lambda_val = self.demand_model.get((time_slot, hex_id), 0)
            if lambda_val <= 0: continue

            # 泊松采样
            num_requests = np.random.poisson(lambda_val * self.scaling_factor)
            if num_requests == 0: continue

            trans_data = self.transition_model.get((time_slot, hex_id))
            if not trans_data: continue

            dests = np.random.choice(trans_data[0], size=num_requests, p=trans_data[1])

            for dest_hex in dests:
                step_second = self.time_step * 60
                props = self.trip_props_model.get((hex_id, dest_hex), {'trip_time': step_second, 'trip_miles': 5})

                dist = props.get('trip_miles', 5)
                dur = props.get('trip_time', step_second)//step_second

                all_orders.append({
                    'origin_hex': hex_id,
                    'dest_hex': dest_hex,
                    'distance': dist,
                    'duration': dur
                })


        return all_orders

    def _init_fixed_orders(self):
        # 初始化固定订单
        if self.fixed_orders is None:
            self.reset_fixed_orders()

    def get_fixed_orders(self, time_slot):
        return self.fixed_orders[time_slot]

    def reset_fixed_orders(self, seed=42):
        # 初始化固定订单
        random.seed(seed)
        np.random.seed(seed)
        self.fixed_orders = [self.generate_dynamic_orders(t) for t in range(self.num_steps)]
        random.seed(None)
        np.random.seed(None)

def get_day_type(date_obj, us_holidays):
    """根据日期判断类型：Holiday, Weekend, Weekday"""
    if date_obj in us_holidays:
        return 'Holiday'
    elif date_obj.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return 'Weekend'
    else:
        return 'Weekday'

def split_dataset(df_all_trips, adjacency):
    print("--- 1. 初始化 HexGridProcessor 并处理全量数据 ---")

    # 执行处理：
    # 1. 生成网格 2. 桥接孤岛 3. 建立邻接表 4. 映射行程数据到网格

    # 此时 df_all_trips 应该包含 parquet 中的所有列，包括时间
    time_col = 'pickup_datetime'
    if df_all_trips[time_col].dtype == 'object':
        df_all_trips[time_col] = pd.to_datetime(df_all_trips[time_col])


    # --- 2. 标记日期类型 ---
    print("\n--- 2. 划分日期类型 (Weekday/Weekend/Holiday) ---")
    us_holidays = holidays.US(years=df_all_trips[time_col].dt.year.unique())

    df_all_trips['date_obj'] = df_all_trips[time_col].dt.date
    df_all_trips['day_type'] = df_all_trips['date_obj'].apply(lambda x: get_day_type(x, us_holidays))

    # --- 3. 分组建立生成器并保存 ---
    os.makedirs(CONFIG['GENETATE_SAVE_DIR'], exist_ok=True)
    day_types = ['Weekday', 'Weekend', 'Holiday']
    sim_list= {day:None for day in day_types}

    for dtype in day_types:
        # 筛选对应类型的数据
        sub_df = df_all_trips[df_all_trips['day_type'] == dtype].copy()

        if len(sub_df) == 0:
            print(f"警告: {dtype} 类型没有数据，跳过。")
            continue

        # 计算该类型实际包含的天数
        num_days = sub_df['date_obj'].nunique()
        print(f"\n正在处理 [{dtype}]:")
        print(f"  - 包含天数: {num_days}")
        print(f"  - 订单总数: {len(sub_df)}")

        # 计算缩放因子 (Scaling Factor)
        # 公式：(仿真司机数 * 单车日均单量) / 真实数据的日均单量
        # 目的：将庞大的真实数据缩放到 CONFIG['N_DRIVERS'] 个司机能承载的量级
        real_daily_avg = len(sub_df) / num_days
        sim_capacity = CONFIG['N_DRIVERS'] * CONFIG['TRIPS_PER_DRIVER_DAY']
        scaling_factor = sim_capacity / real_daily_avg
        print(f"  - 计算缩放因子: {scaling_factor:.4f}")

        # 实例化 PassengerSimulator
        sim = PassengerSimulator(
            df_gridded_trips=sub_df,
            adjacency=adjacency,
            scaling_factor=scaling_factor
        )

        # 保存
        filename = f"simulator_driver_nums={CONFIG['N_DRIVERS']}_hex_scaling={scaling_factor}_{dtype.lower()}.pkl"
        save_path = os.path.join(CONFIG['GENETATE_SAVE_DIR'], filename)
        with open(save_path, 'wb') as f:
            pickle.dump(sim, f)
        print(f"  - 已保存生成器至: {save_path}")
        sim_list[dtype]=sim

    print("\n所有生成器已生成完毕！")
    return sim_list

if __name__ == '__main__':
    # 获得生成器
    SHAPEFILE_PATH = 'taxi_zones/taxi_zones.shp'
    TRIP_DATA_PATH = 'fhvhv_tripdata_2024-01.parquet'  # 需确保文件存在
    HEX_RES = 7

    processor = HexGridProcessor(SHAPEFILE_PATH, TRIP_DATA_PATH, HEX_RES)
    df_trips, gdf_hex = processor.process()
    print(f"全量数据处理完成，共 {len(df_trips)} 条行程。")
    print(f"网格邻接关系数量: {len(processor.adjacency)}")

    # 保存订单生成器
    simulator_list = split_dataset(df_trips, processor.adjacency)

    # 生成测试订单
    # simulator = pickle.load(open('generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl', 'rb'))
    # orders = simulator.generate_dynamic_orders(time_slot=10)
    # fixed_orders=simulator.get_fixed_orders(time_slot=10)
    # print(f"Generated {len(fixed_orders)} orders for hour 10.")

    # 简单绘图
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gdf_hex.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5)
    # processor.gdf_zones.boundary.plot(ax=ax, color='gray', alpha=0.5)
    # ax.set_title(f"Processed Hex Grid (Res {HEX_RES})")
    # plt.savefig('hex_grid_final.png')
    # print("Map saved.")