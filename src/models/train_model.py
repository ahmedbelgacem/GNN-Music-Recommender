import torch
import numpy as np

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm

from utils.sampling import sample_negative_edges


class PlainData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return 0


class SpotifyDataset(Dataset):

    def __init__(self, root, edge_index, transform=None, pre_transform=None, pre_filter=None):
        self.edge_index = edge_index
        self.unique_playlists = torch.unique(edge_index[0, :]).tolist()
        self.num_nodes = len(self.unique_playlists)
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        return self.num_nodes

    def get(self, idx):
        edge_index = self.edge_index[:, self.edge_index[0, :] == idx]  # Return all outgoing edges
        return PlainData(edge_index=edge_index)  # Maybe PlainData IDK


def train(model, data, optimizer, num_playlists, k, n_epochs, batch_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_nodes = data.num_nodes
    split = RandomLinkSplit(num_val=.15, num_test=.15, is_undirected=True, add_negative_train_samples=False,
                            neg_sampling_ratio=0)
    train_split, val_split, test_split = split(data)
    train_ev = SpotifyDataset('root', edge_index=train_split.edge_label_index)  # 'root' is a random string
    train_mp = Data(edge_index=train_split.edge_index)
    val_ev = SpotifyDataset('root', edge_index=val_split.edge_label_index)
    val_mp = Data(edge_index=val_split.edge_index)
    train_loader = DataLoader(train_ev, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ev, batch_size=batch_size, shuffle=False)

    running_loss = 0.
    total_edges = 0
    train_mp = train_mp.to(device)
    val_mp = val_mp.to(device)

    for epoch in range(1, n_epochs + 1):

        model.train()  # Training
        with tqdm(train_loader, unit='batch',
                  desc='Epoch [{:>2}/{:>2}] - {:>30}'.format(epoch, n_epochs, 'Training')) as tbatch:
            for i, batch in enumerate(tbatch):
                optimizer.zero_grad()
                negatives = sample_negative_edges(batch, num_playlists, num_nodes)
                batch, negatives = batch.to(device), negatives.to(device)
                pos_rankings = model(train_mp.edge_index, batch.edge_index)  # Positive samples
                neg_rankings = model(train_mp.edge_index, negatives.edge_index)  # Negative samples
                loss = model.recommendation_loss(pos_rankings, neg_rankings)
                loss.backward()
                optimizer.step()
                # Gather and report
                running_loss += loss.item() * pos_rankings.size(0) * batch.edge_index.shape[1]
                total_edges += batch.edge_index.shape[1]
                if i == len(tbatch) - 1:
                    avg_loss = running_loss / total_edges
                    tbatch.set_postfix_str('Training loss = {:.5f}'.format(avg_loss))

        model.eval()  # Validation
        recalls_dict = {}
        with torch.no_grad():
            with tqdm(val_loader, unit='batch',
                      desc='Epoch [{:>2}/{:>2}] - {:>30}'.format(epoch, n_epochs, 'Validation')) as vbatch:
                for j, batch in enumerate(vbatch):
                    batch = batch.to(device)
                    top_k = model.recommend(val_mp.edge_index, torch.unique(batch.edge_index[0, :]),
                                            torch.unique(batch.edge_index[1, :]), k)
                    unique_playlists = torch.unique(batch.edge_index[0, :])
                    for i, pidx in enumerate(unique_playlists):
                        ground_truth = val_mp.edge_index[1, val_mp.edge_index[0, :] == pidx].cpu()
                        preds = top_k[i].cpu()
                        recall = len(np.intersect1d(ground_truth, preds)) / len(ground_truth)
                        recalls_dict[pidx] = recall
                    if j == len(vbatch) - 1:
                        vbatch.set_postfix_str(
                            'Validation recall = {:.5f}'.format(np.mean(list(recalls_dict.values()))))

    return avg_loss
