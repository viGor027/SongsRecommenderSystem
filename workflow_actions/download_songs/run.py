from workflow_actions.download_songs.source.ncs_scraper import NCSScraper
from workflow_actions.json_handlers import read_json_to_dict

if __name__ == "__main__":
    scraper_config = read_json_to_dict('download_songs_config.json')
    scraper = NCSScraper(**scraper_config)
    scraper.scrape()
