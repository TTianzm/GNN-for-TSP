import numpy as np
from torch_geometric.data import Data
import torch
from itertools import combinations
import matplotlib.pyplot as plt

from gnn_guided_ortools import gnn_edge_pruned_ortools_tsp

plt.rcParams['font.family'] = 'SimHei'      # 设置中文字体为黑体（SimHei）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
from decode_beamsearch import beam_search_decode, two_opt_decode
from model import TSPGNN
from decode_beamsearch import beam_search_decode
from utils import plot_path, plot_edge_heatmap

city_coords = {
    '北京': (116.4074, 39.9042),
    '天津': (117.3616, 39.3434),
    '上海': (121.4737, 31.2304),
    '重庆': (106.5516, 29.5630),
    '香港': (114.1694, 22.3193),
    '澳门': (113.5439, 22.1987),
    '石家庄': (114.5149, 38.0428),
    '太原': (112.5489, 37.8706),
    '呼和浩特': (111.7492, 40.8426),
    '沈阳': (123.4315, 41.8057),
    '长春': (125.3235, 43.8171),
    '哈尔滨': (126.5349, 45.8038),
    '南京': (118.7969, 32.0603),
    '杭州': (120.1551, 30.2741),
    '合肥': (117.2272, 31.8206),
    '福州': (119.2965, 26.0745),
    '南昌': (115.8582, 28.6829),
    '济南': (117.1201, 36.6512),
    '郑州': (113.6254, 34.7466),
    '武汉': (114.3055, 30.5928),
    '长沙': (112.9388, 28.2282),
    '广州': (113.2644, 23.1291),
    '南宁': (108.3669, 22.8170),
    '海口': (110.1983, 20.0440),
    '成都': (104.0668, 30.5728),
    '贵阳': (106.6302, 26.6476),
    '昆明': (102.7189, 25.0389),
    '拉萨': (91.1721, 29.6525),
    '西安': (108.9398, 34.3416),
    '兰州': (103.8343, 36.0611),
    '西宁': (101.7782, 36.6171),
    '银川': (106.2309, 38.4872),
    '乌鲁木齐': (87.6168, 43.8256),
    '台北': (121.5654, 25.0329),
}

def normalize_coords(coord_dict):
    coords = np.array(list(coord_dict.values()))
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    normalized = (coords - min_vals) / (max_vals - min_vals)
    return normalized, list(coord_dict.keys())


def build_tsp_graph(coords):
    coords = torch.tensor(coords, dtype=torch.float)  # [N, 2]
    edge_index = []
    edge_attr = []

    for i, j in combinations(range(len(coords)), 2):
        dist = torch.norm(coords[i] - coords[j])
        edge_index += [[i, j], [j, i]]
        edge_attr += [[dist], [dist]]

    edge_index = torch.tensor(edge_index).T.contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr)


def predict_tsp_path(data, model_path='tsp_gnn_model.pt', beam_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TSPGNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        edge_logits = model(data)
        edge_probs = torch.sigmoid(edge_logits)

    # 可视化边概率热图
    plot_edge_heatmap(data.cpu(), edge_probs.cpu(), title='Edge Probabilities for China Map')

    # 解码路径
    path = beam_search_decode(data.cpu(), edge_probs.cpu(), beam_size=beam_size)

    return path, edge_probs

def plot_china_path(data, path, city_names):
    coords = data.x.cpu().numpy()
    path_coords = coords[path]

    plt.figure(figsize=(10, 10))
    plt.plot(path_coords[:, 0], path_coords[:, 1], '-o', color='green', linewidth=2, markersize=4)

    for i, idx in enumerate(path):
        x, y = coords[idx]
        plt.text(x, y, city_names[idx], fontsize=10, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title('Predicted TSP Tour over Chinese Capital Cities')
    plt.axis('equal')
    #plt.savefig("output.png", transparent=True, dpi=300)
    plt.show()

def compute_path_score(path, edge_index, edge_probs):
    edge_dict = {}
    edge_index = edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()
    for idx, (u, v) in enumerate(edge_index):
        edge_dict[(u, v)] = edge_probs[idx]
        edge_dict[(v, u)] = edge_probs[idx]
    return sum(edge_dict.get((path[i], path[i+1]), 0) for i in range(len(path)-1))


if __name__ == '__main__':
    # 1. 原始城市坐标
    coords_dict = city_coords

    # 2. 归一化 + 获取城市顺序
    normalized_coords, city_names = normalize_coords(coords_dict)

    # 3. 构图
    data = build_tsp_graph(normalized_coords)

    # 4. 预测路径（Beam Search）
    path_beam, edge_probs = predict_tsp_path(data)
    #plot_china_path(data, path_beam, city_names)

    # 5. 2-opt 优化路径
    path_2opt = two_opt_decode(data.cpu(), edge_probs.cpu(), init_path=path_beam)
    #plot_china_path(data, path_2opt, city_names)

    # 6. 打印路径得分对比
    score_beam = compute_path_score(path_beam, data.edge_index, edge_probs)
    score_2opt = compute_path_score(path_2opt, data.edge_index, edge_probs)

    print(f"Beam Search Path Score: {score_beam:.4f}")
    print(f"2-Opt Optimized Path Score: {score_2opt:.4f}")
    path_ortools = gnn_edge_pruned_ortools_tsp(data, edge_probs, keep_ratio=0.2)
    if path_ortools:
        #plot_path(data, path_ortools, title="ILP + GNN Pruned Optimal Path")
        plot_china_path(data, path_ortools, city_names)
