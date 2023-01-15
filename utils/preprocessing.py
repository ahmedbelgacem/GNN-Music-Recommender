import os
import json


def count_playlists(path: str, min_followers: int):
    """
    Checks the number of playlists having more followers than min_followers.

        Parameters:
            path(str): path to the playlists folder.
            min_followers(int): minimum number of followers needed for a playlist to be counted.

        Returns:
            count(int): number of playlists having more than min_followers.
    """
    count = 0
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            playlists = json.load(f)['playlists']
            for playlist in playlists:
                if playlist['num_followers'] > min_followers:
                    count += 1
    return count
