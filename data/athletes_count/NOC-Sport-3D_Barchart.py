import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
medal_data = pd.read_csv('summerOly_medal_counts.csv')  # 金牌数文件
athlete_data = pd.read_csv('summerOly_athletes.csv')  # 参赛数据文件

# 确保数据有必要的列
if not {'Gold', 'NOC', 'Year'}.issubset(medal_data.columns):
    raise ValueError("金牌数据文件缺少必要字段（Gold, NOC, Year）！")

if not {'NOC', 'Sport', 'Year'}.issubset(athlete_data.columns):
    raise ValueError("运动员数据文件缺少必要字段（NOC, Sport, Year）！")

# 1. 找到获得金牌数最多的前10个NOC
top_10_nocs = medal_data.groupby('NOC')['Gold'].sum().nlargest(10).index

# 仅保留这10个NOC的数据，并筛选指定年份的数据
filtered_athlete_data = athlete_data[
    (athlete_data['NOC'].isin(top_10_nocs)) & 
    (athlete_data['Year'].isin([1896, 1928, 1960, 1992, 2024]))
]

# 准备绘图的X、Y轴变量
x_labels = top_10_nocs  # X轴是前10名的NOC
y_labels = filtered_athlete_data['Sport'].unique()  # Y轴是所有的运动类别
years = [1896, 1928, 1960, 1992, 2024]  # 选取的年份
colors = ['r', 'g', 'b', 'orange', 'purple']  # 年份的颜色

# 建立X轴和Y轴的索引字典
x_pos_dict = {noc: i for i, noc in enumerate(x_labels)}
y_pos_dict = {sport: i for i, sport in enumerate(y_labels)}

# 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 2. 遍历年份，统计参赛人数，并绘制柱状图
for year, color in zip(years, colors):
    year_data = filtered_athlete_data[filtered_athlete_data['Year'] == year]
    
    # 使用 pivot_table 统计每个 NOC 和 Sport 的参赛人数
    result = year_data.pivot_table(index='NOC', columns='Sport', aggfunc='size', fill_value=0)
    
    for noc in x_labels:
        for sport in y_labels:
            # 取得对应的参赛人数
            z_count = result.loc[noc, sport] if sport in result.columns and noc in result.index else 0
            # 如果人数为0，则跳过
            if z_count == 0:
                continue
            # 获取x轴和y轴的位置索引
            x_pos = x_pos_dict[noc]
            y_pos = y_pos_dict[sport]
            # 绘制3D柱状条
            ax.bar3d(x_pos, y_pos, 0, 0.5, 0.5, z_count, color=color, alpha=0.8)

# 3. 设置轴标签和图例
ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_xlabel('NOC (Top 10)')

ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels)
ax.set_ylabel('Sport')

ax.set_zlabel('Number of Athletes')

# 图例
legend_patches = [
    plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10) 
    for color in colors
]
ax.legend(legend_patches, years, title="Years", loc='upper left')

# 显示图像
plt.title('Top 10 NOC Participation in Sports Over Selected Years')
plt.tight_layout()
plt.show()
