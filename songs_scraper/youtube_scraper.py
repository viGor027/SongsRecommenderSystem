import os
import yt_dlp
import json

import scraper_settings as settings

# Import videos urls and descriptions from the channel
try:
    with yt_dlp.YoutubeDL(settings.YDL_CHANNEL_OPTS) as ydl:
        result = ydl.extract_info(settings.CHANNEL_URL)

except Exception as e:
    settings.LOGGER.error(e)
    settings.LOGGER.info('An error occurred. Trying to force generic extractor')

    settings.YDL_CHANNEL_OPTS['force_generic_extractor'] = True
    settings.YDL_VIDEO_OPTS['force_generic_extractor'] = True

    try:
        with yt_dlp.YoutubeDL(settings.YDL_CHANNEL_OPTS) as ydl:
            result = ydl.extract_info(settings.CHANNEL_URL)

    except Exception as e:
        settings.LOGGER.error(e)
        settings.LOGGER.critical('An error occurred, exiting')
        print('An error occurred, exiting')
        exit(1)

videos_urls = [video['url'] for video in result['entries']]



def process_url(url: str) -> None:
    """
    Downloads the video and extracts the genre and mood from the description

    Args:
        url (str): The url of the video to process

    Side effects:
        creates a json file with the genres and moods for the video
        creates a mp3 file for the video
    """
    with yt_dlp.YoutubeDL(settings.YDL_VIDEO_OPTS) as ydl:
        # Extracting descriptions
        description = ydl.extract_info(url, download=False)['description']
        description = description.lower()
        description = description.split('\n')
        description = [line for line in description if 'genre and mood' in line]

        if not description:
            settings.LOGGER.warning(f'No genre found in video {url}')
            return
    
        description = description[0].split(':')[-1].strip()
        description = description.split('+')
        description[0] = description[0].split('&')
        description[1] = description[1].split('&')
        description[0] = [genre.strip() for genre in description[0]]
        description[1] = [mood.strip() for mood in description[1]]

        # Saving moods and genres
        video_name = ydl.extract_info(url, download=False)['title']
        video_name = settings.replace_special_chars(video_name)

        file_path = os.path.join(settings.MOODS_GENRES_PATH, f'{video_name}.json')

        with open(file_path, 'w') as file:
            json.dump({'genres': description[0], 'mood': description[1]}, file)

        # Saving audio
        ydl.download([url])


settings.LOGGER.info('Starting processing videos')

for i, url in enumerate(videos_urls):
    print(f'\nProcessing video {i+1}/{len(videos_urls)}')
    process_url(url)

settings.LOGGER.info('All videos processed successfully')







