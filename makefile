ifneq (,$(wildcard .env))
	include .env
	export
endif

install:
	python3 -m venv venv
	. venv/bin/activate; pip install -r requirements.txt

data/raw:
	mkdir -p data/raw
    
download-dataset: init data/raw/$(DATASET_FILE)

data/raw/$(DATASET_FILE): data/raw
	aicrowd login --api-key $(API_KEY)
	aicrowd dataset download --challenge spotify-million-playlist-dataset-challenge 0 -o data/raw

data/processed:
	mkdir -p data/processed

preprocess: data/processed/$(GRAPH_FILE) data/processed/$(PLAYLISTS_FILE)

data/processed/$(GRAPH_FILE): data/processed download-dataset 
	unzip -qq ./data/raw/spotify_million_playlist_dataset.zip
	python3 src/preprocessing/preprocessing.py --source data/raw --out data --min_followers $min_followers

data/processed/$(PLAYLISTS_FILE): data/processed download-dataset 
	unzip -qq ./data/raw/spotify_million_playlist_dataset.zip
	python3 src/preprocessing/preprocessing.py --source data/raw --out data --min_followers $min_followers

train:
	python3 src/models/__main__.py
