import os
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
from pathlib import Path

def read_tsp_file(file_path):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('output')
            coords_str, path_str = parts[0], parts[1]
            coords = list(map(float, coords_str.strip().split()))
            path = list(map(int, path_str.strip().split()))
            samples.append((coords, path))
    return samples

def create_graph(coords, path):
    num_nodes = len(coords) // 2
    nodes = np.array(coords).reshape(num_nodes, 2)  # shape: (n, 2)

    # Build full graph (fully connected)
    edge_index = []
    edge_attr = []
    edge_label = []

    # Build ground truth edge set from path (1-indexed to 0-indexed)
    path_edges = set()
    for i in range(len(path) - 1):
        u = path[i] - 1
        v = path[i + 1] - 1
        path_edges.add((u, v))
        path_edges.add((v, u))  # Undirected

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            edge_index.append([i, j])
            dist = np.linalg.norm(nodes[i] - nodes[j])
            edge_attr.append([dist])
            label = 1 if (i, j) in path_edges else 0
            edge_label.append(label)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)                   # [E, 1]
    edge_label = torch.tensor(edge_label, dtype=torch.float)                # [E]

    data = Data(
        x=torch.tensor(nodes, dtype=torch.float),         # [N, 2]
        edge_index=edge_index,                            # [2, E]
        edge_attr=edge_attr,                              # [E, 1]
        y=edge_label                                      # [E]
    )

    return data

def process_file(file_path, save_path):
    samples = read_tsp_file(file_path)
    data_list = []

    for coords, path in tqdm(samples, desc=f"Processing {file_path}"):
        data = create_graph(coords, path)
        data_list.append(data)

    torch.save(data_list, save_path)
    print(f"Saved {len(data_list)} samples to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='tsp-data')
    parser.add_argument('--output_dir', type=str, default='processed')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for file_name in os.listdir(input_dir):
        if not file_name.endswith('.txt'):
            continue

        file_path = input_dir / file_name

        # Extract tsp size and split name
        base = file_name.replace('.txt', '')  # e.g., tsp10_train_concorde
        save_path = output_dir / (base + '.pt')

        process_file(file_path, save_path)

if __name__ == '__main__':
    main()
