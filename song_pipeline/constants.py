import os

# Paths
PROJECT_FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))
SONGS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'music')
TAGS_DIR = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'moods_genres')

# FeatureExtractor constants
N_MELS = 12
N_SECONDS = 5
SPEC_TYPE = 'mel'
