from utils import *



# 1. 加载数据
file_path = 'data/summerOly_athletes_total_with_types.csv'
data = pd.read_csv(file_path)

# 2. 数据预处理
# 获取独特的运动项目、年份和代表队
sports = sorted(data['Type'].unique())  # 运动项目 S
years = sorted(data['Year'].unique())   # 奥运会年份 T
nocs = sorted(data['NOC'].unique())     # 国家代表队 N

# 构建映射表
sport_to_idx = {sport: i for i, sport in enumerate(sports)}
year_to_idx = {year: i for i, year in enumerate(years)}
noc_to_idx = {noc: i for i, noc in enumerate(nocs)}
S, T, N = len(sports), len(years), len(nocs)




# ============================================================================================
# 重复实验10次
for iteration in range(4,10):
    print(f'====================== Iteration {iteration + 1} ======================')
    # =======================================================================================
    # 训练每类运动对应模型
    for idx in range(S):  

        # 加载平滑后的数据转换
        x = torch.tensor(np.load(f'data/medals_dist_smoothed/{idx}.npy'), dtype=torch.float32)
        # x = torch.tensor(np.load('data/olympic_medals.npy')[idx], dtype=torch.float32)

        T, N = x.size()  # 时间序列长度和向量维度

        # 实例化模型
        model = WeightOptimizer(T)

        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练循环
        num_epochs = 10000
        for epoch in range(num_epochs):
            optimizer.zero_grad()  # 清除梯度
            loss = model(x)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重参数
            # if (epoch + 1) % 100 == 0:
            #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        # 查看学习到的权重
        weights = model.weights
        weights = weights.detach().numpy()
        # 保存学习到的权重
        np.save(f'data/weights/{idx}_No-{iteration}.npy', weights)

        print(f"Learned weights for {idx}-th sport")



    # ===================================================================================
    # ===================================================================================

    for target in [31]:
        # 测试模型，使用2028以前的数据预测2028的国家奖牌总数（即所有运动的奖牌分布之和）
        pred_medals = np.zeros(N)

        for idx in range(S):
            # 加载平滑后的数据转换
            x = torch.tensor(np.load(f'data/medals_dist_smoothed/{idx}.npy'), dtype=torch.float32)
            # 转换为NumPy数组
            x = x.detach().numpy()

            # 加载学习到的权重
            weights = np.load(f'data/weights/{idx}_No-{iteration}.npy')
            weights[-1] = weights[-2] 

            # 乘以距离权重
            dist = torch.tensor([i for i in range(target, 0, -1)], dtype=torch.float32)  # (k,)
            # 通过 inverse_sigmoid 将距离映射为一个单调递减的权重序列
            dist_weights = inverse_sigmoid(dist).detach().numpy()  # (k,)

            # 最终权重为原始权重和距离权重的乘积
            k_weights = dist_weights * weights[:target]  # (k,)
            k_weights = k_weights / np.sum(k_weights)
            # 计算预测值
            x_pred = np.sum(k_weights.reshape(-1,1) * x[:target], axis=0)

            # 将预测值累加到总预测奖牌数
            pred_medals += x_pred 


        # rescale 
        total_medals_list = np.load('data/total_medals_list.npy')
        pred_medals = pred_medals / sum(pred_medals) * (total_medals_list[year_to_idx[2024]] + event_diff)

        # 保存预测结果
        np.save(f'data/result/{target}-pred_medals_{iteration}.npy', pred_medals)
