import json
import torch
from torch_geometric.nn.models import LightGCN
import config as cfg
from src.models.train_model import train

data = torch.load(cfg.GRAPH)

with open(cfg.PLAYLISTS, 'r') as f:
    playlists = json.load(f)
    NUMBER_OF_PLAYLISTS = len(playlists)

model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=cfg.EMBEDDING_DIM,
    num_layers=cfg.NUM_LAYERS
)

if __name__ == '__main__':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.LEARNING_RATE)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train(model, data, optimizer, NUMBER_OF_PLAYLISTS, cfg.K, cfg.EPOCHS, cfg.BATCH_SIZE)