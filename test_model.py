import torch
from torch_geometric.loader import DataLoader
from model import TSPGNN
from decode_beamsearch import beam_search_decode
from utils import plot_path, plot_edge_heatmap
import argparse
import os


@torch.no_grad()
def evaluate(model, dataloader, device, beam_size=30, visualize=True, save_dir="test_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(dataloader):
        data = data.to(device)
        output = model(data)
        probs = torch.sigmoid(output)

        # 可选：热图
        if visualize and i < 3:
            plot_edge_heatmap(data.cpu(), probs.cpu(), title=f"Edge Heatmap Sample {i}",
                              save_path=os.path.join(save_dir, f"heatmap_{i}.png"))

        # beam search 路径
        path = beam_search_decode(data.cpu(), probs.cpu(), beam_size=beam_size)

        # 可视化路径
        if visualize and i < 3:
            plot_path(data.cpu(), path, title=f"Predicted Path (Beam={beam_size})",
                      save_path=os.path.join(save_dir, f"path_pred_{i}.png"))

        # GT 路径可视化
        gt_path = data.y.cpu().tolist()

        # 在 evaluate 函数中添加（在 gt_path = data.y.cpu().tolist() 之后）
        print("\n=== data.y 调试信息 ===")
        print(f"数据类型: {type(data.y)}")  # 检查Tensor类型
        print(f"设备位置: {data.y.device}")  # 检查在CPU/GPU上
        print(f"形状(shape): {data.y.shape}")  # 检查维度
        print(f"内容示例(前10个元素): {data.y.cpu().numpy()[:10]}")  # 检查内容
        print(f"唯一值: {torch.unique(data.y)}")  # 检查包含哪些值
        print(f"是否为整数: {data.y.dtype in (torch.int32, torch.int64)}")  # 检查类型
        print("=====================\n")

        gt_path = data.y.cpu().tolist()

        if visualize and i < 3:
            plot_path(data.cpu(), gt_path, title="Ground Truth Path",
                      save_path=os.path.join(save_dir, f"path_gt_{i}.png"))

        # 输出基本信息
        print(
            f"Sample {i} - Path Length: {len(path)} | Closed: {path[0] == path[-1]} | Unique: {len(set(path)) == len(data.x)}")

        if i >= 10:  # 限制前10个样本
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tsp_gnn_model30.pt")
    parser.add_argument("--data_path", type=str, default="processed/tsp30_test_concorde.pt")
    parser.add_argument("--beam_size", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = TSPGNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # 加载数据
    dataset = torch.load(args.data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluate(model, loader, device, beam_size=args.beam_size, visualize=args.visualize)
