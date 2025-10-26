from workflow_actions.download_songs.source.ncs_scraper import NCSScraper
from workflow_actions.json_handlers import read_json_to_dict
from workflow_actions.paths import DOWNLOAD_SONGS_CONFIG_PATH

scraper_config = read_json_to_dict(DOWNLOAD_SONGS_CONFIG_PATH)
scraper = NCSScraper(**scraper_config)
scraper.scrape()
