from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from shapely.ops import transform


class HexGridProcessor:
    """ 一个将地理空间数据和行程数据处理成六边形网格格式的工具。 """

    def __init__(self, shapefile_path, trip_data_path, hex_resolution):
        """ 初始化处理器。 """
        self.shapefile_path = shapefile_path
        self.trip_data_path = trip_data_path
        self.hex_resolution = hex_resolution
        self.gdf_zones = None
        self.location_centroids = None  # 存储区域中心点
        print(f"--- Initializing HexGridProcessor with H3 Resolution: {self.hex_resolution} ---")

    def _generate_hex_grid(self):
        self.gdf_zones = gpd.read_file(self.shapefile_path)

        # 1. 确保 CRS 是 WGS84
        if self.gdf_zones.crs != 'EPSG:4326':
            self.gdf_zones = self.gdf_zones.to_crs('EPSG:4326')
            print("Reprojected shapefile to EPSG:4326")

        # 2. ⚠️ 必须在此处调用！在生成六边形前创建 centroid 映射
        self._create_location_centroids_map()  # ← 这行绝对不能遗漏！

        # 2. 获取统一多边形
        try:
            unified_polygon = self.gdf_zones.geometry.union_all()
        except:
            unified_polygon = self.gdf_zones.geometry.unary_union

        # 3. 生成六边形
        hex_ids = set()
        if isinstance(unified_polygon, MultiPolygon):
            polys = list(unified_polygon.geoms)
        else:
            polys = [unified_polygon]

        def flip_coords(x, y):
            return y, x

        for poly in polys:
            # outer_lnglat = list(poly.exterior.coords)
            # outer_latlng = [(y, x) for x, y in outer_lnglat]  # (lat, lng)
            #
            # holes_latlng = []
            # for interior in poly.interiors:
            #     hole_lnglat = list(interior.coords)
            #     hole_latlng = [(y, x) for x, y in hole_lnglat]
            #     holes_latlng.append(hole_latlng)

            try:
                # h3_poly = h3.Polygon(outer_latlng, holes_latlng)
                # correct_poly = transform(flip_coords, poly)
                h3_poly = h3.geo_to_h3shape(poly)
                cells = h3.polygon_to_cells(h3_poly, self.hex_resolution)
                hex_ids.update(cells)
            except Exception as e:
                print(f"Error processing polygon: {e}")

        print(f"Generated {len(hex_ids)} hexagons")

        # 4. 创建 GeoDataFrame
        hex_polygons = []
        for h in hex_ids:
            try:
                boundary_latlng = h3.cell_to_boundary(h)
                boundary_lnglat = [(lng, lat) for lat, lng in boundary_latlng]
                hex_polygons.append(Polygon(boundary_lnglat))
            except Exception as e:
                print(f"Failed to create polygon for hex {h}: {e}")
                continue

        self.gdf_hex = gpd.GeoDataFrame(
            data={'hex_id': list(hex_ids)},
            geometry=hex_polygons,
            crs="EPSG:4326"
        )

        # 5. 清理无效几何
        self.gdf_hex = self.gdf_hex[self.gdf_hex.geometry.is_valid]
        self.gdf_hex = self.gdf_hex[self.gdf_hex.geometry.is_empty == False]
        print(f"Valid hexagons after cleanup: {len(self.gdf_hex)}")
        print(f"DEBUG: location_centroids type={type(self.location_centroids)}, "
              f"len={len(self.location_centroids) if self.location_centroids else 'N/A'}")

    def _create_location_centroids_map(self):
        """创建区域ID到中心点经纬度的映射"""
        self.location_centroids = {}  # ⚠️ 关键：必须在方法开头初始化！
        print("   Creating location ID to centroid mapping...")

        id_column = 'LocationID'

        # 计算每个区域的中心点
        self.gdf_zones['centroid'] = self.gdf_zones.geometry.centroid
        self.location_centroids = {}
        id_cnt=Counter ()
        for _, row in self.gdf_zones.iterrows():
            loc_id = row[id_column]
            id_cnt[loc_id]+=1

            # 获取中心点经纬度
            centroid = row['centroid']
            # 注意：Shapely的x是经度，y是纬度
            self.location_centroids[loc_id] = (centroid.x, centroid.y)

        print(f"✅ Created centroid map for {len(self.location_centroids)} zones")

    def _map_trips_to_hex(self):
        """ 将CSV中的行程数据映射到六边形网格上。 """
        print("2. Mapping trip data from CSV to hex grid...")

        print(f"Using centroid map with {len(self.location_centroids)} entries")

        df_trips = pd.read_csv(self.trip_data_path)

        # 确保LocationID是字符串类型
        df_trips['PULocationID'] = df_trips['PULocationID']
        df_trips['DOLocationID'] = df_trips['DOLocationID']

        # 获取中心点函数
        def get_centroid(loc_id):
            if loc_id in self.location_centroids:
                return self.location_centroids[loc_id]
            return None, None

        # 添加中心点列
        df_trips['pickup_centroid'] = df_trips['PULocationID'].apply(get_centroid)
        df_trips['dropoff_centroid'] = df_trips['DOLocationID'].apply(get_centroid)

        # 分离经纬度
        df_trips['pickup_longitude'] = df_trips['pickup_centroid'].apply(lambda x: x[0] if x else None)
        df_trips['pickup_latitude'] = df_trips['pickup_centroid'].apply(lambda x: x[1] if x else None)
        df_trips['dropoff_longitude'] = df_trips['dropoff_centroid'].apply(lambda x: x[0] if x else None)
        df_trips['dropoff_latitude'] = df_trips['dropoff_centroid'].apply(lambda x: x[1] if x else None)

        # 转换为六边形ID
        def lat_lng_to_hex(lat, lng):
            if pd.isna(lat) or pd.isna(lng) or lat is None or lng is None:
                return None
            try:
                return h3.latlng_to_cell(lat, lng, self.hex_resolution)
            except Exception as e:
                print(f"Error converting ({lat}, {lng}): {str(e)}")
                return None

        df_trips['pickup_hex_id'] = df_trips.apply(
            lambda row: lat_lng_to_hex(row['pickup_latitude'], row['pickup_longitude']), axis=1
        )
        df_trips['dropoff_hex_id'] = df_trips.apply(
            lambda row: lat_lng_to_hex(row['dropoff_latitude'], row['dropoff_longitude']), axis=1
        )

        # 清理中间列
        df_trips.drop(columns=['pickup_centroid', 'dropoff_centroid'], inplace=True)

        # 删除无效的映射
        initial_count = len(df_trips)
        df_trips.dropna(subset=['pickup_hex_id', 'dropoff_hex_id'], inplace=True)
        mapped_count = len(df_trips)

        print(f" Successfully mapped {mapped_count} out of {initial_count} trips to hex grid.")
        print(f" Mapping success rate: {mapped_count / initial_count:.2%}")
        return df_trips

    def process(self):
        self._generate_hex_grid()
        df_gridded_trips = self._map_trips_to_hex()
        print("--- HexGridProcessor finished. ---")
        return df_gridded_trips, self.gdf_hex


class PassengerSimulator:
    def __init__(self, df_gridded_trips, scaling_factor=1.0):
        self.df = df_gridded_trips
        # 尝试解析时间列 - 检查可用的日期时间列
        datetime_cols = ['pickup_datetime', 'request_datetime', 'on_scene_datetime']
        datetime_col = None

        for col in datetime_cols:
            if col in self.df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            raise ValueError(f"No datetime column found. Available columns: {self.df.columns.tolist()}")

        print(f"Using '{datetime_col}' as datetime column for simulation")
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        self.df['hour'] = self.df[datetime_col].dt.hour

        self.scaling_factor = scaling_factor
        self.demand_model = {}
        self.transition_model = {}
        self.trip_props_model = {}
        self._learn_distributions()

    def _learn_distributions(self):
        print("--- Learning distributions from gridded trip data ---")
        # 计算唯一日期数量
        datetime_cols = ['pickup_datetime', 'request_datetime', 'on_scene_datetime']
        datetime_col = next(col for col in datetime_cols if col in self.df.columns)

        num_days = self.df[datetime_col].dt.date.nunique() or 1
        demand_counts = self.df.groupby(['hour', 'pickup_hex_id']).size() / num_days
        self.demand_model = demand_counts.to_dict()

        transitions = self.df.groupby(['hour', 'pickup_hex_id', 'dropoff_hex_id']).size().reset_index(name='count')
        for (hour, origin), group in transitions.groupby(['hour', 'pickup_hex_id']):
            total = group['count'].sum()
            if total > 0:
                self.transition_model[(hour, origin)] = (
                    group['dropoff_hex_id'].values,
                    group['count'].values / total
                )

        # 使用数据集中存在的列
        distance_col = 'trip_distance' if 'trip_distance' in self.df.columns else 'trip_miles'
        duration_col = 'trip_duration' if 'trip_duration' in self.df.columns else 'trip_time'

        props = self.df.groupby(['pickup_hex_id', 'dropoff_hex_id'])[[distance_col, duration_col]].mean()
        self.trip_props_model = props.to_dict('index')
        print("--- Learning complete. ---")

    def generate_orders(self, time_slot, all_hex_ids):
        all_orders = []
        for hex_id in all_hex_ids:
            lambda_val = self.demand_model.get((time_slot, hex_id), 0)
            if lambda_val <= 0:
                continue

            num_potential_requests = np.random.poisson(lambda_val * self.scaling_factor)
            if num_potential_requests == 0:
                continue

            for _ in range(num_potential_requests):
                transition_data = self.transition_model.get((time_slot, hex_id))
                if not transition_data or len(transition_data[0]) == 0:
                    continue

                dest_hex = np.random.choice(transition_data[0], p=transition_data[1])
                props = self.trip_props_model.get((hex_id, dest_hex),
                                                  {(
                                                       'trip_distance' if 'trip_distance' in self.df.columns else 'trip_miles'): 1.0,
                                                   (
                                                       'trip_duration' if 'trip_duration' in self.df.columns else 'trip_time'): 5})

                # 获取正确的列名
                distance_col = 'trip_distance' if 'trip_distance' in props else 'trip_miles'
                duration_col = 'trip_duration' if 'trip_duration' in props else 'trip_time'

                order = {
                    'origin_hex': hex_id,
                    'dest_hex': dest_hex,
                    'distance': props[distance_col],
                    'duration': props[duration_col],
                }
                all_orders.append(order)
        return all_orders


if __name__ == '__main__':
    # --- 配置 ---
    SHAPEFILE_PATH = 'taxi_zones/taxi_zones.shp'
    TRIP_DATA_PATH = 'fhvhv_jan_01.csv'
    HEX_RESOLUTION = 6

    # 1. 处理数据
    processor = HexGridProcessor(SHAPEFILE_PATH, TRIP_DATA_PATH, HEX_RESOLUTION)
    df_gridded_trips, gdf_hex = processor.process()

    # 2. 构建乘客模拟器
    simulator = PassengerSimulator(df_gridded_trips, scaling_factor=0.1)

    # 3. 演示生成订单
    print("\n--- Demonstrating order generation for time_slot=10 ---")
    all_hex_ids = gdf_hex['hex_id'].tolist()
    generated_orders = simulator.generate_orders(time_slot=10, all_hex_ids=all_hex_ids)
    print(f"Generated {len(generated_orders)} orders.")
    if generated_orders:
        print("Example order:", generated_orders[0])

    # 4. 可视化验证（增强版）
    print("\n--- Generating visualization map... ---")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 1️⃣ 先绘制出租车区域边界（作为底图）
    processor.gdf_zones.boundary.plot(
        ax=ax,
        color='gray',
        linewidth=1,
        alpha=0.7,
        aspect='auto'  # 关键：提前设置纵横比避免后续计算
    )

    # 2️⃣ 严格清理六边形网格（三重保险）
    gdf_hex_clean = gdf_hex.copy()
    gdf_hex_clean = gdf_hex_clean[gdf_hex_clean.geometry.notnull()]
    gdf_hex_clean = gdf_hex_clean[gdf_hex_clean.geometry.is_valid]
    gdf_hex_clean = gdf_hex_clean[gdf_hex_clean.geometry.is_empty]
    gdf_hex_clean = gdf_hex_clean.reset_index(drop=True)

    print(f"Cleaned hex grid: {len(gdf_hex_clean)} valid hexagons (was {len(gdf_hex)})")

    # 3️⃣ 安全绘制六边形网格（带异常回退）
    try:
        # 方法1：优先使用 geopandas 绘制（带 aspect='auto'）
        gdf_hex_clean.boundary.plot(
            ax=ax,
            color='blue',
            linewidth=0.5,
            alpha=0.5,
            aspect='auto'  # 关键参数：跳过危险的纵横比计算
        )
    except Exception as e1:
        print(f"Warning: GeoPandas plot failed ({e1}), falling back to manual plotting...")
        # 方法2：手动用 matplotlib 绘制（绝对安全）
        for geom in gdf_hex_clean.geometry:
            try:
                if geom.is_valid and not geom.is_empty:
                    x, y = geom.exterior.xy
                    # 检查坐标有效性
                    if all(np.isfinite(x)) and all(np.isfinite(y)):
                        ax.plot(x, y, color='blue', linewidth=0.5, alpha=0.5)
            except Exception:
                continue
        ax.set_aspect('auto')  # 手动设置纵横比

    # 4️⃣ 安全绘制行程点（过滤无效坐标）
    sample_size = min(1000, len(df_gridded_trips))
    if sample_size > 0:
        sample_trips = df_gridded_trips.sample(n=sample_size)
        # 严格过滤无效坐标
        valid_mask = (
                sample_trips['pickup_longitude'].notnull() &
                sample_trips['pickup_latitude'].notnull() &
                np.isfinite(sample_trips['pickup_longitude']) &
                np.isfinite(sample_trips['pickup_latitude'])
        )
        valid_trips = sample_trips[valid_mask]

        if len(valid_trips) > 0:
            ax.scatter(
                valid_trips['pickup_longitude'],
                valid_trips['pickup_latitude'],
                s=2,
                c='red',
                alpha=0.6,
                label=f'Pickup Locations ({len(valid_trips)})'
            )
            print(f"Plotted {len(valid_trips)} valid pickup points")
        else:
            print("Warning: No valid pickup coordinates found for visualization")

    # 5️⃣ 最终设置（关键！）
    ax.set_title(f'Hex Grid (Res {HEX_RESOLUTION}) over NYC Taxi Zones', fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('auto')  # 最终保险：强制设置
    plt.tight_layout()

    # 保存并尝试显示
    try:
        plt.savefig('hex_grid_map.png', dpi=150, bbox_inches='tight')
        print("✅ Map successfully saved to 'hex_grid_map.png'")
    except Exception as e:
        print(f"Warning: Failed to save figure: {e}")

    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not display plot interactively ({e}), but file was saved.")