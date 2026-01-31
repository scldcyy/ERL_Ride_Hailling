import pandas as pd

# 读取CSV文件
# 假设文件名为 dataset.csv
df = pd.read_csv('fhvhv_jan_01.csv')

# 方法1：使用 value_counts() 直接统计每个 PULocationID 的出现次数
# 这会自动按计数降序排列
location_counts = df['PULocationID'].value_counts().reset_index()
location_counts.columns = ['PULocationID', 'count']
location_counts=location_counts.sort_values(by='PULocationID', ascending=True)

print("分组统计结果：")
print(location_counts)

# 方法2：使用 groupby() 进行分组并计数 (如果需要对其他列进行聚合，这种方式更通用)
# grouped_counts = df.groupby('PULocationID').size().reset_index(name='count')
# print(grouped_counts)
