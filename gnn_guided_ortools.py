from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import torch

def gnn_edge_pruned_ortools_tsp(data, edge_probs, keep_ratio=0.2):
    """
    使用 GNN 输出筛选高概率边，构造 OR-Tools 输入，并求解最优 TSP 路径
    """
    edge_index = data.edge_index.cpu().numpy()
    edge_probs = edge_probs.detach().cpu().numpy()

    n = data.num_nodes
    coords = data.x.cpu().numpy()

    # Step 1: 构建稀疏邻接矩阵，只保留高概率边
    E = len(edge_probs)
    num_keep = int(E * keep_ratio)
    keep_indices = edge_probs.argsort()[-num_keep:]  # top-k 边索引

    distance_matrix = np.full((n, n), np.inf)

    for idx in keep_indices:
        u, v = edge_index[:, idx]
        dist = np.linalg.norm(coords[u] - coords[v])
        distance_matrix[u][v] = dist
        distance_matrix[v][u] = dist

    # OR-Tools 要求整数距离
    distance_matrix[np.isinf(distance_matrix)] = 1e6  # 不可达设大值
    distance_matrix = np.round(distance_matrix * 10000).astype(int)

    # Step 2: 构建 OR-Tools TSP 路由器
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 设置启发式初始解策略（可选）
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(5)

    # Step 3: 求解
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        path = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            path.append(node)
            index = solution.Value(routing.NextVar(index))
        path.append(path[0])  # 闭合成环

        return path
    else:
        print("❌ OR-Tools 未能求解 TSP")
        return None
