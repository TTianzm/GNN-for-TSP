import numpy as np
import torch
import heapq
from collections import namedtuple
import heapq

BeamState = namedtuple('BeamState', ['path', 'score', 'visited'])

def beam_search_decode(data, edge_probs, beam_size=5, max_len=None):
    """
    data: PyG graph data (with .edge_index)
    edge_probs: Tensor [E] - edge probabilities (sigmoid output)
    Returns: best_path (list of node indices)
    """
    edge_dict = {}
    edge_index = data.edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()

    # 构造边表: edge_dict[from][to] = score
    for idx, (u, v) in enumerate(edge_index):
        if u not in edge_dict:
            edge_dict[u] = {}
        if v not in edge_dict:
            edge_dict[v] = {}
        edge_dict[u][v] = edge_probs[idx]
        edge_dict[v][u] = edge_probs[idx]  # TSP是无向图

    n = data.num_nodes
    max_len = n if max_len is None else max_len

    beams = [BeamState(path=[start], score=0.0, visited=set([start])) for start in range(n)]

    for step in range(1, max_len):
        new_beams = []
        for beam in beams:
            last = beam.path[-1]
            for nei, prob in edge_dict.get(last, {}).items():
                if nei in beam.visited:
                    continue
                new_path = beam.path + [nei]
                new_score = beam.score + prob  # 累加 logit 可视为路径得分
                new_visited = set(beam.visited)
                new_visited.add(nei)
                new_beams.append(BeamState(new_path, new_score, new_visited))

        # 取 top-k
        beams = heapq.nlargest(beam_size, new_beams, key=lambda b: b.score)

    # 让路径回到起点（形成环）
    final_paths = []
    for beam in beams:
        if len(beam.path) == n:
            last = beam.path[-1]
            start = beam.path[0]
            closing_score = edge_dict[last].get(start, 0.0)
            total_score = beam.score + closing_score
            final_paths.append((beam.path + [start], total_score))

    if not final_paths:
        print("⚠️ 未能生成完整路径")
        return beams[0].path  # fallback

    # 选择得分最高的路径
    final_paths.sort(key=lambda x: x[1], reverse=True)
    return final_paths[0][0]

def two_opt_decode(data, edge_probs, init_path=None, max_iter=10000):
    """
    对初始路径应用 2-opt 优化，提升边概率总和
    edge_probs: [E]，每条边的概率（sigmoid 输出）
    init_path: 初始路径（列表），若为 None，则默认顺序
    """
    # 构建边查找表
    edge_dict = {}
    edge_index = data.edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()

    for idx, (u, v) in enumerate(edge_index):
        edge_dict[(u, v)] = edge_probs[idx]
        edge_dict[(v, u)] = edge_probs[idx]

    n = data.num_nodes
    if init_path is None:
        path = list(range(n)) + [0]
    else:
        path = init_path.copy()

    def path_score(p):
        return sum(edge_dict.get((p[i], p[i+1]), 0) for i in range(len(p)-1))

    best_score = path_score(path)

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_score = path_score(new_path)
                if new_score > best_score:
                    path = new_path
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return path

def branch_and_bound_decode(data, edge_probs, upper_bound_path=None, max_nodes=1e6):
    """
    使用分支定界算法搜索 TSP 最优路径（使用 GNN 边概率引导分支）
    - edge_probs: Tensor [E]，边概率
    - upper_bound_path: 可选，作为初始上界的路径（如 Beam Search 输出）
    - 返回：得分最高路径
    """
    edge_index = data.edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()
    n = data.num_nodes

    # 构建邻接表
    edge_dict = {}
    for idx, (u, v) in enumerate(edge_index):
        if u not in edge_dict:
            edge_dict[u] = {}
        edge_dict[u][v] = edge_probs[idx]
        if v not in edge_dict:
            edge_dict[v] = {}
        edge_dict[v][u] = edge_probs[idx]

    # 初始上界：使用 Beam Search 提供的路径
    def path_score(path):
        return sum(edge_dict[path[i]][path[i+1]] for i in range(len(path) - 1))

    if upper_bound_path is not None:
        best_path = upper_bound_path
        best_score = path_score(upper_bound_path)
    else:
        best_path = None
        best_score = -np.inf

    # 初始队列（每项：(-估计总分, 当前路径, 已访问节点, 当前得分)）
    heap = []
    for start in range(n):
        heapq.heappush(heap, (-0.0, [start], set([start]), 0.0))

    steps = 0

    while heap and steps < max_nodes:
        est_neg_score, path, visited, score = heapq.heappop(heap)
        last = path[-1]

        if len(path) == n:
            # 闭合路径
            if path[0] in edge_dict[last]:
                total_score = score + edge_dict[last][path[0]]
                full_path = path + [path[0]]
                if total_score > best_score:
                    best_score = total_score
                    best_path = full_path
            continue

        # 获取当前点未访问邻居，按边概率降序排列（分支顺序引导）
        neighbors = [(nei, edge_dict[last][nei]) for nei in edge_dict[last] if nei not in visited]
        neighbors.sort(key=lambda x: -x[1])  # 高概率优先扩展

        for nei, prob in neighbors:
            new_path = path + [nei]
            new_score = score + prob
            new_visited = set(visited)
            new_visited.add(nei)

            # 估计未来分数（保守估计用最大边）
            remaining = n - len(new_path)
            optimistic_remaining = remaining * prob  # 使用当前边概率作为估计上限
            upper_estimate = new_score + optimistic_remaining

            if upper_estimate < best_score:
                continue  # 剪枝

            heapq.heappush(heap, (-upper_estimate, new_path, new_visited, new_score))
        steps += 1

    if best_path is None:
        print("⚠️ 分支定界未能找到完整路径")
        return path
    return best_path
