import json
import torch
from torch_geometric.nn.models import LightGCN
from config import GRAPH, PLAYLISTS, EPOCHS, EMBEDDING_DIM, BATCH_SIZE, NUM_LAYERS, K, LEARNING_RATE, MODELS_FOLDER
from src.models.train_model import train

data = torch.load(GRAPH)

with open(PLAYLISTS, 'r') as f:
    playlists = json.load(f)
    NUMBER_OF_PLAYLISTS = len(playlists)

model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS
)

if __name__ == '__main__':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train(model, data, optimizer, NUMBER_OF_PLAYLISTS, K, EPOCHS, BATCH_SIZE)
    torch.save(model, MODELS_FOLDER)