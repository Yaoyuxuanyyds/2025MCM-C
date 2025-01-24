import pandas as pd

# 读取CSV文件
data = pd.read_csv('summerOly_athletes.csv')

# 确保数据中有Year, NOC, Sport三个字段
if not {'Year', 'NOC', 'Sport'}.issubset(data.columns):
    raise ValueError("输入表格中缺少必须的Year, NOC或Sport字段！")

# 获取所有独特的NOC作为完整的行集合
all_nocs = data['NOC'].unique()

# 循环处理每一年的数据
years = data['Year'].unique()  # 获取所有独特的年份
for year in years:
    # 筛选出对应年份的数据
    year_data = data[data['Year'] == year]

    # 统计每个NOC参加每种Sport的人数总和
    result = year_data.pivot_table(index='NOC', columns='Sport', aggfunc='size', fill_value=0)
    
    # 确保所有的NOC行都存在，补充缺失的行为0
    result = result.reindex(index=all_nocs, fill_value=0)

    # 保存结果到新的CSV文件
    output_filename = f'{year}.csv'
    result.to_csv(output_filename)
    print(f"已生成 {output_filename} 文件。")