import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import TSPGNN
import os

def load_dataset(path, batch_size=32):
    dataset = torch.load(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)        # [num_edges]
        target = batch.y           # [num_edges]
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            loss = F.binary_cross_entropy_with_logits(pred, batch.y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 修改此处路径 ===
    train_path = 'processed/tsp30_train_concorde.pt'
    val_path   = 'processed/tsp30_val_concorde.pt'

    train_loader = load_dataset(train_path, batch_size=64)
    val_loader   = load_dataset(val_path, batch_size=64)

    model = TSPGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'tsp_gnn_model30.pt')
    print("✅ 模型已保存为 tsp_gnn_model30.pt")

    # ✅ 绘制训练曲线
    from utils import plot_training_curves
    plot_training_curves(train_losses, val_losses,save_path = "training_curve.png")

if __name__ == '__main__':
    main()
