import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('summerOly_athletes.csv')

# 确保数据中有 Year, NOC, Sport 三个字段
if not {'Year', 'NOC', 'Sport'}.issubset(data.columns):
    raise ValueError("输入表格中缺少必须的 Year, NOC 或 Sport 字段！")

# 获取所有唯一的 Sport、NOC 和 Year
all_sports = data['Sport'].unique()  # 不同的 Sport
all_years = data['Year'].unique()    # 不同的 Year
all_nocs = data['NOC'].unique()      # 不同的 NOC

# 循环处理每种 Sport
for sport in all_sports:
    # 筛选出当前 Sport 的数据
    sport_data = data[data['Sport'] == sport]
    
    # 使用 pivot_table 统计每个 NOC 在不同年份的参赛人数
    result = sport_data.pivot_table(
        index='NOC',  # 纵坐标
        columns='Year',  # 横坐标
        aggfunc='size',  # 按人数统计
        fill_value=0     # 没有参赛时用 0 填充
    )
    
    # 确保每个 Sport 表格包含所有的 NOC 和 Year
    # 补全所有 NOC
    result = result.reindex(all_nocs, fill_value=0)
    # 补全所有 Year
    result = result.reindex(columns=all_years, fill_value=0)
    
    # 保存结果到 CSV 文件
    output_filename = f'{sport}.csv'
    result.to_csv(output_filename)
    print(f"已生成文件：{output_filename}")

print("全部处理完毕！🎉")
