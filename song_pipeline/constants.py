import os

# Paths
PROJECT_FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))
SONGS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'music')
TAGS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'moods_genres')
LABELS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'labels')
DATA_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'data')

# FeatureExtractor constants
N_MELS = 80
N_SECONDS = 5
STEP = 3
SPEC_TYPE = 'mel'
