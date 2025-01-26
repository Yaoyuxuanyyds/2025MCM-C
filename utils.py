import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 定义权重优化模型
class WeightOptimizer(nn.Module):
    def __init__(self, T):
        super(WeightOptimizer, self).__init__()
        # 初始化全局权重序列为可训练参数
        self.weights = nn.Parameter(torch.rand(T, requires_grad=True))  # 长度为 T 的权重序列

    def inverse_sigmoid(self, x, k=-0.7, c=20):
        """
        将1到30的整数映射到0到1范围内，并呈现反sigmoid函数的单调递减趋势。
        :param x: 输入值 (torch.Tensor)
        :param k: 控制曲线陡峭程度
        :param c: 控制曲线中心点
        :return: 映射到0到1范围内的值
        """
        return 1 / (1 + torch.exp(k * ((31 - x) - c)))

    def forward(self, x):
        T, N = x.size()  # 时间序列长度和向量维度
        total_loss = 0  # 损失累计
        Start = 1  # 从时间步 15 开始预测
        for k in range(Start, T):  # 从时间步 15 开始预测
            # 定义一个时间距离序列，从k到1
            dist = torch.tensor([i for i in range(k, 0, -1)], dtype=torch.float32, device=x.device)  # (k,)
            # 通过 inverse_sigmoid 将距离映射为一个单调递减的权重序列
            dist_weights = self.inverse_sigmoid(dist)  # (k,)
            # 最终权重为原始权重和距离权重的乘积
            k_weights = dist_weights * self.weights[:k]  # (k,)

            # 根据权重加权计算预测值
            x_pred = torch.sum(k_weights.unsqueeze(1) * x[:k], dim=0) / torch.sum(k_weights)  # (N,)
            # 计算当前时间步的预测损失
            total_loss += torch.norm(x_pred - x[k]) ** 2

        # 返回平均损失
        return total_loss / (T - Start)




def smooth_distribution(A, B, smooth_factor=0.1, reduce_factor=0.1):
    """
    改进的平滑函数。

    参数:
    - A: 原始分布 (list or np.array)
    - B: 参考分布 (list or np.array)
    - smooth_factor: 控制 B 对平滑值的影响程度 (默认 0.1)
    - reduce_factor: 控制 A[i] 的减少量占比 (默认 0.1)

    返回:
    - A_smoothed: 平滑后的分布 (np.array)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    # 确保 A 和 B 的长度相等
    assert A.shape == B.shape, f"A({A.shape}) and B({B.shape}) must have the same shape."
    # 标准化 B，避免 B 全零的情况
    if B.sum() > 0:
        B_norm = B / B.sum()
    else:
        B_norm = np.zeros_like(B)
    
    # 构造平滑因子
    S = B_norm * smooth_factor
    # 初始化平滑后的 A
    A_smoothed = np.zeros_like(A)
    
    # 对 A 的每一项进行更新
    for i in range(len(A)):
        if A[i] == 0:
            # A[i] 为 0 的情况下，平滑值来源于 S[i]
            A_smoothed[i] = S[i]
        else:
            # A[i] 为非零时，减少量与 S[i] 成反比，减少的比例由 reduce_factor 控制
            decrease = reduce_factor * A[i] / (1 + S[i])
            A_smoothed[i] = A[i] - decrease
    
    
    return A_smoothed


# 统计结果
event_diff = 10
name_to_noc = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "ALG", "Argentina": "ARG", "Armenia": "ARM",
    "Australasia": "ANZ", "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE", "Bahamas": "BAH",
    "Bahrain": "BRN", "Barbados": "BAR", "Belarus": "BLR", "Belgium": "BEL", "Bermuda": "BER",
    "Bohemia": "BOH", "Botswana": "BOT", "Brazil": "BRA", "British West Indies": "BWI", "Bulgaria": "BUL",
    "Burkina Faso": "BUR", "Burundi": "BDI", "Cabo Verde": "CPV", "Cameroon": "CMR", "Canada": "CAN",
    "Ceylon": "CEY", "Chile": "CHI", "China": "CHN", "Chinese Taipei": "TPE", "Colombia": "COL",
    "Costa Rica": "CRC", "Croatia": "CRO", "Cuba": "CUB", "Cyprus": "CYP", "Czech Republic": "CZE",
    "Czechoslovakia": "TCH", "Denmark": "DEN", "Djibouti": "DJI", "Dominica": "DMA",
    "Dominican Republic": "DOM", "East Germany": "GDR", "Ecuador": "ECU", "Egypt": "EGY",
    "Eritrea": "ERI", "Estonia": "EST", "Ethiopia": "ETH", "FR Yugoslavia": "YUG", "Fiji": "FIJ",
    "Finland": "FIN", "Formosa": "TWN", "France": "FRA", "Gabon": "GAB", "Georgia": "GEO",
    "Germany": "GER", "Ghana": "GHA", "Great Britain": "GBR", "Greece": "GRE", "Grenada": "GRN",
    "Guatemala": "GUA", "Guyana": "GUY", "Haiti": "HAI", "Hong Kong": "HKG", "Hungary": "HUN",
    "Iceland": "ISL", "Independent Olympic Athletes": "IOA",
    "Independent Olympic Participants": "IOP", "India": "IND", "Indonesia": "INA", "Iran": "IRI",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Ivory Coast": "CIV",
    "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Kosovo": "KOS", "Kuwait": "KUW", "Kyrgyzstan": "KGZ", "Latvia": "LAT", "Lebanon": "LIB",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Macedonia": "MKD", "Malaysia": "MAS",
    "Mauritius": "MRI", "Mexico": "MEX", "Mixed team": "ZZX", "Moldova": "MDA", "Mongolia": "MGL",
    "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ", "Namibia": "NAM",
    "Netherlands": "NED", "Netherlands Antilles": "AHO", "New Zealand": "NZL", "Niger": "NIG",
    "Nigeria": "NGR", "North Korea": "PRK", "North Macedonia": "MKD", "Norway": "NOR",
    "Pakistan": "PAK", "Panama": "PAN", "Paraguay": "PAR", "Peru": "PER", "Philippines": "PHI",
    "Poland": "POL", "Portugal": "POR", "Puerto Rico": "PUR", "Qatar": "QAT", "ROC": "ROC",
    "Refugee Olympic Team": "ROT", "Romania": "ROU", "Russia": "RUS", "Russian Empire": "RU1",
    "Saint Lucia": "LCA", "Samoa": "SAM", "San Marino": "SMR", "Saudi Arabia": "KSA",
    "Senegal": "SEN", "Serbia": "SRB", "Serbia and Montenegro": "SCG", "Singapore": "SGP",
    "Slovakia": "SVK", "Slovenia": "SLO", "South Africa": "RSA", "South Korea": "KOR",
    "Soviet Union": "URS", "Spain": "ESP", "Sri Lanka": "SRI", "Sudan": "SUD",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "SUI", "Syria": "SYR", "Taiwan": "TPE",
    "Tajikistan": "TJK", "Tanzania": "TAN", "Thailand": "THA", "Togo": "TOG", "Tonga": "TGA",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR", "Turkmenistan": "TKM",
    "Uganda": "UGA", "Ukraine": "UKR", "Unified Team": "EUN", "United Arab Emirates": "UAE",
    "United States": "USA", "United Team of Germany": "EUA", "Uruguay": "URU", "Uzbekistan": "UZB",
    "Venezuela": "VEN", "Vietnam": "VIE", "Virgin Islands": "ISV", "West Germany": "FRG",
    "Yugoslavia": "YUG", "Zambia": "ZAM", "Zimbabwe": "ZIM"
}

def inverse_sigmoid(x, k=-0.7, c=20):
    """
    将1到30的整数映射到0到1范围内，并呈现反sigmoid函数的单调递减趋势。
    :param x: 输入值 (torch.Tensor)
    :param k: 控制曲线陡峭程度
    :param c: 控制曲线中心点
    :return: 映射到0到1范围内的值
    """
    return 1 / (1 + torch.exp(k * ((31 - x) - c)))

