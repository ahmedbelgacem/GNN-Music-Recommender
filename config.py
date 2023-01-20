import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).parent
PROCESSED_FOLDER = ROOT / 'data/processed'
MODELS_FOLDER = ROOT / 'models'
GRAPH = PROCESSED_FOLDER / 'graph.pt'
PLAYLISTS = PROCESSED_FOLDER / 'playlists.json'

NECESSARY_DATA_FILES = (GRAPH, PLAYLISTS)

for necessary_file in NECESSARY_DATA_FILES:
    if not necessary_file.exists():
        raise Exception(f"File '{necessary_file.name}' is needed in folder {PROCESSED_FOLDER}")

mandatory_env_variables = (
    "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "K", "NUM_LAYERS", "EMBEDDING_DIM")

for variable in mandatory_env_variables:
    if not os.getenv(variable, None):
        raise Exception(
            f"Environment variable '{variable}' is missing.\n"
            f"Make sure you've added it to the .env file"
        )

EPOCHS = int(os.getenv("EPOCHS", None))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", None))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", None))
K = int(os.getenv("K", None))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", None))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", None))
