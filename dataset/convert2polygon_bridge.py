from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import networkx as nx


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

        # 2.  必须在此处调用！在生成六边形前创建 centroid 映射
        self._create_location_centroids_map()  # ← 这行绝对不能遗漏！

        # 2. 获取统一多边形
        unified_polygon = self.gdf_zones.geometry.union_all()

        # 3. 生成六边形
        hex_ids = set()
        if isinstance(unified_polygon, MultiPolygon):
            polys = list(unified_polygon.geoms)
        else:
            polys = [unified_polygon]


        for poly in polys:
            h3_poly = h3.geo_to_h3shape(poly)
            cells = h3.polygon_to_cells(h3_poly, self.hex_resolution)
            hex_ids.update(cells)

        print(f"Generated {len(hex_ids)} hexagons")

        # 4. 创建 GeoDataFrame
        hex_polygons = []
        for h in hex_ids:
            boundary_latlng = h3.cell_to_boundary(h)
            boundary_lnglat = [(lng, lat) for lat, lng in boundary_latlng]
            hex_polygons.append(Polygon(boundary_lnglat))

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

        id_column = 'OBJECTID'

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
            return h3.latlng_to_cell(lat, lng, self.hex_resolution)

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

    # 将此方法添加到 HexGridProcessor 类中
    def _bridge_disconnected_components(self, max_bridge_distance=15):
        """
        核心算法：检测孤岛并建立'桥梁'网格以确保连通性。
        :param max_bridge_distance: 允许的最大桥梁长度（六边形数量）。
                                    超过此距离的孤岛（如远海岛屿）将不会被连接。
        """
        print(f"3. Analyzing connectivity and bridging gaps (Max dist: {max_bridge_distance})...")

        # 将当前的 hex_id 集合转换为列表以便操作
        current_hexes = set(self.gdf_hex['hex_id'])

        # 循环直到不再产生新的合并（或者只剩一个连通分量）
        iteration = 0
        while True:
            iteration += 1
            # 1. 构建当前网格的邻接图
            G = nx.Graph()
            for h in current_hexes:
                G.add_node(h)
                # 获取周围6个邻居
                neighbors = h3.grid_ring(h, 1)
                for n in neighbors:
                    if n in current_hexes:
                        G.add_edge(h, n)

            # 2. 获取连通分量 (Components)
            components = list(nx.connected_components(G))
            if len(components) <= 1:
                print("   ✅ Grid is fully connected!")
                break

            print(f"   Iteration {iteration}: Found {len(components)} disconnected islands.")

            # 3. 寻找最近的两个孤岛进行连接 (最小生成树策略的简化版)
            # 我们只需要找到所有分量对中，距离最近的一对，把它们连起来，然后下一轮循环会处理剩下的

            min_dist = float('inf')
            best_bridge = None  # (start_hex, end_hex)

            # 为了性能，我们只计算不同分量之间的距离
            # 优化：对于每个分量，只取其“边界”网格来计算距离会更快，但这里为了代码简洁使用全量计算
            # 考虑到NYC区域网格量级，如果非常慢，需要优化为KDTree或只取边界

            # 这里的策略是：找到主大陆（最大的分量），尝试把其他小分量连到主大陆上
            # 按大小排序，最大的在第一个
            components.sort(key=len, reverse=True)
            main_land = components[0]
            other_islands = components[1:]

            bridges_built_this_round = 0

            # 尝试将每个小岛连接到最近的现有陆地（不一定是主大陆，只要是其他分量即可，但连主大陆最直观）
            # 这里使用简化的贪心策略：每个小岛找到离它最近的“非己方”网格

            # 收集所有非当前岛屿的网格用于距离比对（这在大数据量下需要空间索引优化，这里演示逻辑）
            # 为加速，我们只尝试连接到主大陆
            main_land_list = list(main_land)

            for island in other_islands:
                island_list = list(island)

                # 寻找该岛到主大陆的最短距离
                # 暴力法比较慢，但逻辑最简单。
                # 优化：随机抽样或只计算边缘。
                # 在此演示代码中，我们假设网格数量在几千级别，暴力尚可。
                # 若卡顿，请告知，我会提供 KDTree 优化版本。

                local_min_dist = float('inf')
                local_best_pair = None

                # 简单的剪枝：只检查岛屿边缘和主大陆边缘
                # (这里简化处理，直接遍历)
                for h_island in island_list:
                    for h_main in main_land_list:
                        try:
                            # H3 距离计算很极速
                            d = h3.grid_distance(h_island, h_main)
                            if d < local_min_dist:
                                local_min_dist = d
                                local_best_pair = (h_island, h_main)
                                if d <= 2: break  # 距离够近了，直接连
                        except:
                            continue  # 处理可能的异常
                    if local_min_dist <= 2: break

                # 如果找到可行桥梁
                if local_best_pair and local_min_dist <= max_bridge_distance:
                    start, end = local_best_pair
                    # 生成直线路径
                    try:
                        bridge_path = h3.grid_path_cells(start, end)
                        # 将桥梁加入集合
                        initial_len = len(current_hexes)
                        current_hexes.update(bridge_path)
                        if len(current_hexes) > initial_len:
                            bridges_built_this_round += 1
                            print(f"      Built bridge of length {local_min_dist} between clusters.")
                    except Exception as e:
                        print(f"      Failed to build path: {e}")

            if bridges_built_this_round == 0:
                print("   No more feasible bridges found within max distance.")
                break

        # 更新最终结果
        print(f"   Bridging complete. Total hexes: {len(current_hexes)}")

        # 重新生成 GeoDataFrame
        hex_polygons = []
        for h in current_hexes:
            boundary_latlng = h3.cell_to_boundary(h)
            boundary_lnglat = [(lng, lat) for lat, lng in boundary_latlng]
            hex_polygons.append(Polygon(boundary_lnglat))

        self.gdf_hex = gpd.GeoDataFrame(
            data={'hex_id': list(current_hexes)},
            geometry=hex_polygons,
            crs="EPSG:4326"
        )

    def _build_adjacency_map(self):
        """ 建立六边形网格的邻接关系 (每个网格连接周围6个方向) """
        print("   Building adjacency graph (6-connectivity)...")
        self.adjacency = {}
        # 将所有有效的 hex_id 转为 set 以便快速查找
        valid_hexes = set(self.gdf_hex['hex_id'])

        for hex_id in valid_hexes:
            # 获取该网格周围 1 环的邻居 (不包含自身)
            # h3.grid_ring 返回的是指定距离的环上的六边形
            try:
                neighbors = h3.grid_ring(hex_id, 1)
            except Exception:
                # 处理可能的异常（如五边形边界情况）
                neighbors = []

            # 筛选出实际上存在于我们网格系统中的邻居
            # 如果您希望“逻辑上”连接即使该邻居不在区域内，可去掉 if 判断
            # 但为了保证“互相可达”是基于有效网格的，建议保留过滤
            valid_neighbors = [n for n in neighbors if n in valid_hexes]

            self.adjacency[hex_id] = valid_neighbors

        print(f"✅ Adjacency graph built. Access via self.adjacency[hex_id].")

    def process(self):
        self._generate_hex_grid()  # 使用原始的贴合陆地的生成方式

        # 新增步骤：桥接孤岛
        self._bridge_disconnected_components(max_bridge_distance=3)

        # 重新清理无效几何（以防万一）
        self.gdf_hex = self.gdf_hex[self.gdf_hex.geometry.is_valid]

        # 建立邻接关系
        self._build_adjacency_map()

        df_gridded_trips = self._map_trips_to_hex()
        print("--- HexGridProcessor finished. ---")
        return df_gridded_trips, self.gdf_hex


class PassengerSimulator:
    def __init__(self, df_gridded_trips,adjacency, scaling_factor=1.0):
        self.df = df_gridded_trips
        self.adjacency = adjacency
        # 尝试解析时间列 - 检查可用的日期时间列
        datetime_col = 'pickup_datetime'

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
        datetime_col = 'pickup_datetime'

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


    def real_orders(self):
        tmp=self.df.groupby(['pickup_hex_id']).size().reset_index(name='count')
        tmp=tmp.to_dict('index')
        cnt={v['pickup_hex_id']:v['count'] for k,v in tmp.items()}
        return cnt

    import numpy as np

    def check_cancellation(self, t_waited, current_eta, price, valuation):
        # 1. 设定参数 (这些参数可以通过真实数据拟合，或者在论文中说明是假设的)
        alpha = 0.05  # 基础风险系数
        gamma = 1.5  # 形状参数，>1 表示越等越急

        beta_eta = 0.2  # ETA 对取消的影响系数 (ETA越长越想取消)
        beta_price = 0.01  # 价格影响
        beta_val = -0.02  # 估值影响 (这单越值钱，越舍不得取消，所以是负号)

        # 2. 计算当前的风险率 Hazard Rate
        # 基础风险：随已等待时间(t_waited)指数增长
        baseline_hazard = alpha * (t_waited ** (gamma - 1))

        # 协变量影响：
        covariates = beta_eta * current_eta + beta_price * price + beta_val * valuation

        # 综合风险率
        lambda_t = baseline_hazard * np.exp(covariates)

        # 3. 转化为这一秒取消的概率
        prob_cancel = 1 - np.exp(-lambda_t * 1)  # delta_t = 1 second

        # 4. 伯努利试验
        return np.random.random() < prob_cancel


if __name__ == '__main__':
    # --- 配置 ---
    SHAPEFILE_PATH = 'taxi_zones/taxi_zones.shp'
    TRIP_DATA_PATH = 'fhvhv_jan_01.csv'
    HEX_RESOLUTION = 7

    # 1. 处理数据
    processor = HexGridProcessor(SHAPEFILE_PATH, TRIP_DATA_PATH, HEX_RESOLUTION)
    df_gridded_trips, gdf_hex = processor.process()

    # 2. 构建乘客模拟器
    simulator = PassengerSimulator(df_gridded_trips, processor.adjacency, scaling_factor=1.2)

    # 3. 演示生成订单
    print("\n--- Demonstrating order generation for time_slot=10 ---")
    all_hex_ids = gdf_hex['hex_id'].tolist()
    generated_orders = simulator.generate_orders(time_slot=10, all_hex_ids=all_hex_ids)

    print(f"Generated {len(generated_orders)} orders.")

    # 4. 绘制订单数量热力图
    print("\n--- Generating order count heatmap... ---")
    # a. 统计每个六边形网格的订单数
    order_counts = simulator.real_orders()

    # b. 将统计结果转换为 DataFrame
    df_counts = pd.DataFrame(list(order_counts.items()), columns=['hex_id', 'order_count'])

    # c. 将订单数量合并回 gdf_hex
    gdf_hex_with_orders = gdf_hex.merge(df_counts, on='hex_id', how='left')

    # d. 将没有订单的六边形（NaN）的计数值填充为0
    gdf_hex_with_orders['order_count'].fillna(0, inplace=True)

    # e. 绘制热力图
    fig_heatmap, ax_heatmap = plt.subplots(1, 1, figsize=(15, 12))

    gdf_hex_with_orders.plot(
        column='order_count',
        ax=ax_heatmap,
        legend=True,
        cmap='viridis',
        legend_kwds={'label': "Number of Orders", 'orientation': "horizontal"}
    )

    # 叠加绘制出租车区域边界（作为参考）
    processor.gdf_zones.boundary.plot(
        ax=ax_heatmap,
        color='white',
        linewidth=0.5,
        alpha=0.4
    )

    ax_heatmap.set_title(f'Heatmap of Generated Orders (Time Slot 10, Res {HEX_RESOLUTION})', fontsize=16)
    ax_heatmap.set_xlabel("Longitude")
    ax_heatmap.set_ylabel("Latitude")
    ax_heatmap.grid(False)
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    plt.tight_layout()
    plt.savefig('order_heatmap.png', dpi=200, bbox_inches='tight')
    print("Heatmap successfully saved to 'order_heatmap.png'")
    plt.show()

    # 4. 可视化验证（增强版）
    print("\n--- Generating visualization map... ---")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    processor.gdf_zones.boundary.plot(
        ax=ax,
        color='gray',
        linewidth=1,
        alpha=0.7,
        aspect='auto'  # 关键：提前设置纵横比避免后续计算
    )

    gdf_hex_clean = gdf_hex.copy()
    gdf_hex_clean = gdf_hex_clean[gdf_hex_clean.geometry.notnull()]
    gdf_hex_clean = gdf_hex_clean[gdf_hex_clean.geometry.is_valid]
    gdf_hex_clean = gdf_hex_clean.reset_index(drop=True)

    print(f"Cleaned hex grid: {len(gdf_hex_clean)} valid hexagons (was {len(gdf_hex)})")

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


    ax.set_title(f'Hex Grid (Res {HEX_RESOLUTION}) over NYC Taxi Zones', fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('auto')  # 最终保险：强制设置
    plt.tight_layout()

    # 保存并尝试显示
    plt.savefig('hex_grid_map.png', dpi=150, bbox_inches='tight')
    print("Map successfully saved to 'hex_grid_map.png'")

    plt.show()