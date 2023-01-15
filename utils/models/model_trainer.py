import torch
from torch_geometric.data import Data


def sample_negative_edges(batch, num_playlists, num_nodes):
    # Randomly samples songs for each playlist. Here we sample 1 negative edge
    # for each positive edge in the graph, so we will
    # end up having a balanced 1:1 ratio of positive to negative edges.

    negs = []
    for i in batch.edge_index[0, :]:  # looping over playlists
        assert i < num_playlists  # just ensuring that i is a playlist
        rand = torch.randint(num_playlists, num_nodes, (1,))  # randomly sample a song
        negs.append(rand.item())

    edge_index_negs = torch.row_stack([batch.edge_index[0, :], torch.LongTensor(negs)])
    return Data(edge_index=edge_index_negs)
