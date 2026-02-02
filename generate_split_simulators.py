import pandas as pd
import numpy as np
import holidays
import pickle
import os
import sys

# 确保能导入 dataset 目录下的模块
# 假设当前脚本位于项目根目录，dataset 文件夹在同级
sys.path.append('dataset')

# 导入您指定的 convert2polygon_bridge 中的类
from dataset.convert2polygon_bridge import HexGridProcessor, PassengerSimulator

# --- 配置部分 ---
CONFIG = {
    # 仿真环境参数 (用于计算缩放因子)
    'N_DRIVERS': 1000,  # 仿真环境中的司机数量
    'TRIPS_PER_DRIVER_DAY': 25,  # 期望每个司机每天完成的订单数 (用于控制总单量)

    # 数据与网格配置
    'SHAPEFILE_PATH': 'dataset/taxi_zones/taxi_zones.shp',  # 原始区域形状文件
    'TRIP_DATA_PATH': 'dataset/fhv_tripdata_2025-01.parquet',  # 原始行程数据 (CSV格式)
    'HEX_RESOLUTION': 7,  # H3 网格分辨率
    'SAVE_DIR': 'model/generators'  # 模型保存路径
}


def get_day_type(date_obj, us_holidays):
    """根据日期判断类型：Holiday, Weekend, Weekday"""
    if date_obj in us_holidays:
        return 'Holiday'
    elif date_obj.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return 'Weekend'
    else:
        return 'Weekday'


def main():
    print("--- 1. 初始化 HexGridProcessor 并处理全量数据 ---")
    # 初始化处理器
    processor = HexGridProcessor(
        shapefile_path=CONFIG['SHAPEFILE_PATH'],
        trip_data_path=CONFIG['TRIP_DATA_PATH'],
        hex_resolution=CONFIG['HEX_RESOLUTION']
    )

    # 执行处理：
    # 1. 生成网格 2. 桥接孤岛 3. 建立邻接表 4. 映射行程数据到网格
    # process() 返回的是包含了 pickup_hex_id, dropoff_hex_id 的 DataFrame
    df_all_trips, gdf_hex = processor.process()

    # 此时 df_all_trips 应该包含 CSV 中的所有列，包括时间
    # 确保时间列是 datetime 类型 (convert2polygon_bridge 中可能还没转)
    time_col = 'pickup_datetime'
    if df_all_trips[time_col].dtype == 'object':
        df_all_trips[time_col] = pd.to_datetime(df_all_trips[time_col])

    print(f"全量数据处理完成，共 {len(df_all_trips)} 条行程。")
    print(f"网格邻接关系数量: {len(processor.adjacency)}")

    # --- 2. 标记日期类型 ---
    print("\n--- 2. 划分日期类型 (Weekday/Weekend/Holiday) ---")
    us_holidays = holidays.US(years=df_all_trips[time_col].dt.year.unique())

    df_all_trips['date_obj'] = df_all_trips[time_col].dt.date
    df_all_trips['day_type'] = df_all_trips['date_obj'].apply(lambda x: get_day_type(x, us_holidays))

    # --- 3. 分组建立生成器并保存 ---
    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
    simulators = {}
    day_types = ['Weekday', 'Weekend', 'Holiday']

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
        # 目的：将庞大的真实数据缩放到 50 个司机能承载的量级
        real_daily_avg = len(sub_df) / num_days
        sim_capacity = CONFIG['N_DRIVERS'] * CONFIG['TRIPS_PER_DRIVER_DAY']
        scaling_factor = sim_capacity / real_daily_avg
        print(f"  - 计算缩放因子: {scaling_factor:.4f}")

        # 实例化 PassengerSimulator (使用 convert2polygon_bridge 中的类)
        # 注意：这里传入的是 processor.adjacency，这是该类特有的参数
        sim = PassengerSimulator(
            df_gridded_trips=sub_df,
            adjacency=processor.adjacency,
            scaling_factor=scaling_factor
        )

        # 保存
        filename = f"simulator_hex_{dtype.lower()}.pkl"
        save_path = os.path.join(CONFIG['SAVE_DIR'], filename)
        with open(save_path, 'wb') as f:
            pickle.dump(sim, f)
        print(f"  - 已保存生成器至: {save_path}")

    print("\n所有生成器已生成完毕！")


if __name__ == "__main__":
    main()