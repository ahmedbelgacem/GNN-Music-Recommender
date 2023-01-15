# Building a Spotify Playlist Track Recommender using a Graph Neural Network approch:  
## Goal:  
Automatic playlist continuation: given an initial set of tracks in a playlist, predict the subsequent tracks in that playlist.  
## Dataset:  
We used in this project the dataset from the Spotify Million Playlist Dataset Challenge. The Spotify Million Playlist Dataset Challenge consists of a dataset and evaluation to enable research in music recommendations. It is a continuation of the RecSys Challenge 2018.  
The dataset contains 1,000,000 playlists and over 2 million unique tracks by nearly 300,000 artists including playlist titles and track titles, created by users on the Spotify platform between January 2010 and October 2017. It represents the largest public dataset of music playlists in the world.  
### Model:  
LightGCN is a state-of-the-art Graph Neural Network for collaborative filtering.  
LightGCN learns user-item embeddings by propagation them on the graph and applies a weighted sum to obtain a final embedding of a node (user or item).  

LightGCN has outperformed other GNN recommendation models, namely NGCF. In fact, LightGCN removed all learnable parameters from the traditional NGCF and replaced them with learnable parameters within the shallow embeddings of input nodes.  

## Usage:
Install the requirements.  
You will first need to download the dataset from the aicrowd [plateforme](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)  
You will then need to run the preprocessing script on the dataset to create the graph and the playlists json file.  
Finally run the train_model script to run the lightGCN model.  

## Authors:
[Ahmed Belgacem - Software Engineering graduate from the National Institute of Applied Sciences and Technology (INSAT) and Big Data, Artificial Intelligence master student at Paris Dauphine Tunis.](https://www.linkedin.com/in/ahmedbelgacem/)  
[Nizar Masmoudi - Software Engineering graduate from École supérieure privée d'ingénierie et de technologie (Esprit) and Big Data, Artificial Intelligence master student at Paris Dauphine Tunis.](https://www.linkedin.com/in/nizar-masmoudi/)
