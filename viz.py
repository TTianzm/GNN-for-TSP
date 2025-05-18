import torch
from model import TSPGNN
from utils import plot_edge_heatmap, plot_path
from torch_geometric.utils import to_networkx
import networkx as nx
from decode_beamsearch import beam_search_decode, two_opt_decode
from decode_beamsearch import beam_search_decode, branch_and_bound_decode
from gnn_guided_ortools import gnn_edge_pruned_ortools_tsp


def greedy_decode_edge_probs(data, edge_probs):
    """
    简单贪婪方式从边概率中恢复路径（非最优，但用于可视化）
    """
    import networkx as nx
    G = nx.Graph()
    edge_index = data.edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()

    for (idx, (u, v)) in enumerate(edge_index):
        G.add_edge(u, v, weight=edge_probs[idx])

    # 构造最大权路径（近似TSP解）
    path = list(nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=True))
    return path

def edges_to_path(edge_index, edge_labels):
    """
    从边标签重建完整TSP路径
    参数:
        edge_index: 形状[2, E]的边索引张量
        edge_labels: 形状[E]的边标签张量(0/1)
    返回:
        按顺序排列的节点列表表示的环路路径
    """
    # 创建无向图
    G = nx.Graph()

    # 添加所有在路径中的边(edge_label=1的边)
    edge_list = edge_index.t().tolist()  # 转换为[E, 2]的列表
    for idx, (u, v) in enumerate(edge_list):
        if edge_labels[idx] == 1:
            G.add_edge(u, v)

    # 检查是否为单一环路
    if not nx.is_connected(G):
        raise ValueError("路径边不构成单一环路")

    # 获取欧拉回路(对于TSP就是哈密尔顿环路)
    try:
        path = list(nx.eulerian_circuit(G, source=0))
        # 提取节点顺序(去掉重复的中间节点)
        node_order = [path[0][0]]  # 起始节点
        for u, v in path:
            node_order.append(v)
        return node_order
    except nx.NetworkXError:
        # 如果不能形成欧拉回路，尝试其他方法
        return list(nx.dfs_preorder_nodes(G, source=0)) + [0]

def compute_path_score(path, edge_index, edge_probs):
    edge_dict = {}
    edge_index = edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()
    for idx, (u, v) in enumerate(edge_index):
        edge_dict[(u, v)] = edge_probs[idx]
        edge_dict[(v, u)] = edge_probs[idx]
    return sum(edge_dict.get((path[i], path[i+1]), 0) for i in range(len(path)-1))

def main():
    model = TSPGNN()
    model.load_state_dict(torch.load("tsp_gnn_model30.pt", map_location="cpu"))
    model.eval()

    data_list = torch.load("processed/tsp30_test_concorde.pt")
    data = data_list[2]

    with torch.no_grad():
        pred = model(data)

    # 可视化预测边热图
    plot_edge_heatmap(data, torch.sigmoid(pred), title="Predicted Edge Probabilities")

    edge_index = data.edge_index
    edge_labels = data.y

    try:
        true_path = edges_to_path(edge_index, edge_labels)
        plot_path(data, true_path, title="Ground Truth Path")
    except Exception as e:
        print(f"无法重建真实路径: {e}")
        # 回退方案：显示节点位置但不连接
        plot_path(data, [], title="Node Positions (Path not available)")

    # 可视化模型解码路径
    # Beam Search 解码路径
    edge_probs = torch.sigmoid(pred)
    path_beam = beam_search_decode(data, edge_probs, beam_size=8)
    plot_path(data, path_beam, title="Beam Search Path")

    # 使用 2-opt 进一步优化
    path_2opt = two_opt_decode(data, edge_probs, init_path=path_beam)
    plot_path(data, path_2opt, title="2-Opt Optimized Path")

    path_ortools = gnn_edge_pruned_ortools_tsp(data, edge_probs, keep_ratio=0.4)
    if path_ortools:
        plot_path(data, path_ortools, title="ILP + GNN Pruned Optimal Path")


if __name__ == '__main__':
    main()

