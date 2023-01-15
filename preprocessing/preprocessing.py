"""
    Preprocesses the spotify million playlist dataset to create a graph containing only playlist having a certain number
    of followers along with their songs.
    The script uses networkx to create the graph, then converts it to the Pytorch Geometric Data format and saves it.
    The script also stores two dictionnaries for the mapping from playlists pid or song track_id to graph index.

    Usage:
        python preprocessing.py --source path-to-json-playlists-files --out output-directory-path [--min_followers
        minimum-number-of-followers]
"""
import os
import json
import sys
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from utils.preprocessing import count_playlists


def preprocess(source_folder, output_folder, min_followers):
    playlist_id = 0
    song_id = count_playlists(source_folder, min_followers)
    playlist_dict = {}
    song_dict = {}
    G = nx.Graph()

    with tqdm(os.listdir(source_folder)) as files:
        for file in files:
            with open(os.path.join(source_folder, file), 'r') as f:
                playlists = json.load(f)['playlists']
                for playlist in playlists:
                    if playlist['num_followers'] > min_followers:
                        G.add_node(playlist_id)
                        tmp = {k: v for k, v in playlist.items() if k not in ['tracks', 'num_edits', 'modified_at']}
                        tmp['id'] = playlist_id
                        playlist_dict[playlist['pid']] = tmp
                        for song in playlist['tracks']:
                            track_uri = song['track_uri']
                            if track_uri not in song_dict:
                                G.add_node(song_id)
                                tmp = {k: v for k, v in song.items()}
                                tmp['id'] = song_id
                                song_dict[song['track_uri']] = tmp
                                song_id += 1
                            G.add_edge(playlist_id, song_dict[track_uri]['id'])
                        playlist_id += 1

    edge_list = []
    for edge in G.edges:
        edge_list.append(list(edge))
        edge_list.append(list(edge)[::-1])
    edge_index = torch.LongTensor(edge_list)
    data = Data(edge_index=edge_index.t().contiguous(), num_nodes=len(G.nodes))
    torch.save(data, os.path.join(output_folder, 'graph_object.pt'))
    with open(os.path.join(output_folder, 'playlist_dict.json'), 'w') as f:
        json.dump(playlist_dict, f)
    with open(os.path.join(output_folder, 'song_dict.json'), 'w') as f:
        json.dump(song_dict, f)


def usage():
    print(sys.argv[0],
          '--source path-to-json-playlists-files --out output-directory-path [--min_followers '
          'minimum-number-of-followers]')


if __name__ == "__main__":
    raw_data_folder = None
    output_data_folder = None
    min_followers_number = 0

    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg == '--source':
            raw_data_folder = args.pop(0)
        elif arg == '--out':
            output_data_folder = args.pop(0)
        elif arg == '--min_followers':
            min_followers_number = int(args.pop(0))
    if raw_data_folder and output_data_folder:
        preprocess(raw_data_folder, output_data_folder, min_followers_number)
    else:
        usage()
