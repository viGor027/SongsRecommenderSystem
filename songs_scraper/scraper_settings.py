import logging
import os

ALL_SONGS = False
MAX_SONGS = 3 # ignored if ALL_SONGS is True 
MAX_PAGES = 1 # ignored if ALL_SONGS is True

DOWNLOAD_FILE_PATH = os.path.dirname(os.path.dirname(__file__))
DOWNLOAD_FILE_PATH = os.path.join(DOWNLOAD_FILE_PATH, 'downloads')
MUSIC_PATH = os.path.join(DOWNLOAD_FILE_PATH, 'music')
MOODS_GENRES_PATH = os.path.join(DOWNLOAD_FILE_PATH, 'moods_genres')

os.makedirs(DOWNLOAD_FILE_PATH, exist_ok=True)
os.makedirs(MUSIC_PATH, exist_ok=True)
os.makedirs(MOODS_GENRES_PATH, exist_ok=True)


# TODO: make another file for logging settings
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs.log')

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(pathname)s - %(levelname)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH),  # Logowanie do pliku
        logging.StreamHandler()  # Logowanie do konsoli (stdout)
    ]
)

LOGGER = logging.getLogger(__name__)

# Youtube scraper settings
CHANNEL_URL = 'https://www.youtube.com/c/audiolibrary-channel/videos'

YDL_CHANNEL_OPTS = {
    'quiet': False,
    'extract_flat': True,
    'download': False,
    'playlistend': MAX_SONGS if not ALL_SONGS else None,
}

YDL_VIDEO_OPTS = {
    'format': 'bestaudio/best',
    'quiet': False,
    'extract_flat': True,
    'outtmpl': os.path.join(MUSIC_PATH, '%(title)s.%(ext)s'),
}

# Ncs scraper settings
NCS_URL = 'https://ncs.io/music-search?q=&genre=&mood=&version=regular&page='

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
}

# lambda (i <= settings.MAX_PAGES or settings.ALL_SONGS)
check_pages = lambda i: i <= MAX_PAGES or ALL_SONGS


# Helper functions
def replace_special_chars(text: str) -> str:
    """
    Replaces special characters in the text with underscores

    Args:
        text (str): The text to replace special characters in
    """

    special_chars = ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', '.', ' ', '\n', '\t', '\r', '\b', '\f', '\v', ',', '.']
    for char in special_chars:
        text = text.replace(char, '_')
    return text
