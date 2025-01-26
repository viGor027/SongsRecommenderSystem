import os

# Paths
PROJECT_FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))
SONGS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'music')
TAGS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'moods_genres')
LABELS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'labels')
DATA_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'data')

SPACE_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'space')

# FeatureExtractor constants
N_MELS = 80
N_SECONDS = 5
STEP = 1
SPEC_TYPE = 'mel'
