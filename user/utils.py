import os

SPACE_FOLDER_PATH = os.path.dirname(os.path.dirname(__file__))
SPACE_FOLDER_PATH = os.path.join(SPACE_FOLDER_PATH, "downloads")
SPACE_FOLDER_PATH = os.path.join(SPACE_FOLDER_PATH, "space")

SPACE_FILE_PATH = os.path.join(SPACE_FOLDER_PATH, "space.pt")
SPACE_INDEX_FILE_PATH = os.path.join(SPACE_FOLDER_PATH, "space_index.json")
