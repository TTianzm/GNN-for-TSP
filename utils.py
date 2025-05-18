import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np


def plot_edge_heatmap(data, edge_probs, title='Edge Probabilities', save_path=None):
    G = nx.Graph()
    pos = {i: tuple(coord.tolist()) for i, coord in enumerate(data.x)}

    # 添加节点和带权重的边
    for i in range(data.x.size(0)):
        G.add_node(i)

    edge_index = data.edge_index.t().tolist()
    edge_probs = edge_probs.detach().cpu().numpy()

    for (idx, (u, v)) in enumerate(edge_index):
        G.add_edge(u, v, weight=edge_probs[idx])

    # 绘制图形
    plt.figure(figsize=(6, 6))
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw(G, pos,
            node_color='lightblue',
            with_labels=True,
            edge_color=weights,
            edge_cmap=plt.cm.Reds,
            width=2)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    plt.colorbar(sm)
    plt.title(title)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图形释放内存
    else:
        plt.show()  # 无保存路径时显示


def plot_path(data, path, title="Path", save_path=None):
    coords = data.x.cpu().numpy()
    path = np.array(path, dtype=int)  # ✅ 修复索引问题

    # 创建图形
    plt.figure(figsize=(6, 6))

    if len(path) > 0:  # 如果有路径数据
        path_coords = coords[path]
        # 绘制路径线
        plt.plot(path_coords[:, 0], path_coords[:, 1], '-o', color='green', linewidth=2, markersize=8)
        # 添加节点标签
        for i, (x, y) in enumerate(path_coords):
            plt.text(x, y, str(path[i]), fontsize=10, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    else:  # 如果没有路径数据（仅显示节点）
        plt.scatter(coords[:, 0], coords[:, 1], color='blue', s=50)

    # 绘制所有节点（确保孤立节点可见）
    plt.scatter(coords[:, 0], coords[:, 1], color='lightgray', s=20, alpha=0.5)

    plt.title(title)
    plt.axis('equal')  # 保证x/y轴比例相同

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图形释放内存
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, val_accs=None, save_path=None):
    """
    Args:
        train_losses (list): Training loss values per epoch
        val_losses (list): Validation loss values per epoch
        val_accs (list, optional): Validation accuracy values per epoch
        save_path (str, optional): Path to save the figure (e.g., "training_curves.png")
    """
    epochs = len(train_losses)
    plt.figure(figsize=(10, 4))

    # Plot Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # Plot Accuracy Curve (if provided)
    if val_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), val_accs, label='Val Accuracy', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()

    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")

    plt.show()
