import requests
from bs4 import BeautifulSoup
import json
import os
import time

import scraper_settings as settings

def process_page(page_no: int) -> bool:
    """
    Downloads the mp3 files and extracts the genres and moods from the page

    Args:
        page_no (int): The page number to process

    Returns:
        bool: False if an error occurred, True otherwise

    Side effects:
        creates a json file with the genres and moods for each song
        creates a mp3 file for each song
    """
    response = requests.get(settings.NCS_URL + str(page_no), headers=settings.headers) 

    if response.status_code != 200:
        settings.LOGGER.error(f'Error {response.status_code} while fetching page {page_no}')
        return False
    
    soup = BeautifulSoup(response.text, 'html.parser')
    song_cards = soup.find_all('a', class_='player-play')

    titles = []
    urls = []
    genres = []
    moods = []

    for song_card in song_cards:
        title = song_card['data-track']
        title += ' - '
        title += song_card['data-artistraw']
        title = settings.replace_special_chars(title)

        url = song_card['data-url']
        genre = song_card['data-genre']

        titles.append(title)
        urls.append(url)
        genres.append([genre])

    # Every 4th td contains the mood
    moods = soup.find_all('td', style="width:15%;")
    moods = [card for i, card in enumerate(moods) if i % 4 == 0]
    moods = [card.find_all('a', href=True) for card in moods]
    moods = [[mood.text for mood in card] for card in moods]
    moods = [card[1:] for card in moods]

    for title, genre, mood in zip(titles, genres, moods):
        path = os.path.join(settings.MOODS_GENRES_PATH, f'{title}.json')
        with open(path, 'w') as file:
            json.dump({'genres': genre, 'mood': mood}, file)

    for title, url in zip(titles, urls):
        try:
            mp3_response = requests.get(url, headers=settings.headers)
            if mp3_response.status_code != 200:
                raise Exception(f'Error {mp3_response.status_code} while fetching mp3 {url} for page {page_no}')
        except Exception as e:
            settings.LOGGER.error(e)
            return False
        
        path = os.path.join(settings.MUSIC_PATH, f'{title}.mp3')
        with open(path, 'wb') as file:
            file.write(mp3_response.content)

# Main loop working for all pages
# unless settings.MAX_PAGES is set
settings.LOGGER.info('Starting processing pages')

i = 1
while process_page(i) and settings.check_pages(i):
    settings.LOGGER.info(f'Page {i} processed')
    i += 1
    time.sleep(1)

settings.LOGGER.info('Finished processing pages')

# <a class="player-play" 
# data-artist="&lt;a href='/artist/1190/derek-cate' 
# style='color:#191d24;
# '&gt;Derek Cate&lt;/a&gt;, 
# &lt;
# a href='/artist/1191/b3nte'
# style='color:#191d24;'
# &gt;B3nte&lt;/a&gt;, &lt;a 
# href='/artist/553/mangoo' 
# style='color:#191d24;'
# &gt;Mangoo&lt;/a&gt;" 
# data-artistraw="Derek Cate, B3nte, Mangoo" 
# data-cover="https://ncsmusic.s3.eu-west-1.amazonaws.com/tracks/000/001/798/100x100/1730713312_hsfbBKXzm4_Perfection_FINAL_shrunk.jpg" 
# data-genre="Dance-Pop" 
# data-preview="98" 
# data-tid="9bbbb432-febf-4122-9eb9-070dfca80902" 
# data-track="Perfection (Feat. Derek Cate)" 
# data-url="https://ncsmusic.s3.eu-west-1.amazonaws.com/tracks/000/001/798/1730713313_7UzY0cpVMj_01-Mangoo-B3nte---Perfection-Feat.-Derek-Cate-NCS-Release.mp3" 
# data-versions="Regular">
# <i class="far fa-play-circle"></i></a>]