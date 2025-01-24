import os
import shutil
import math
from song_pipeline.constants import SONGS_DIR


def split_music_files():
    # Define folder paths
    music_folder = SONGS_DIR
    music1_folder = os.path.join(music_folder, 'music1')
    music2_folder = os.path.join(music_folder, 'music2')
    music3_folder = os.path.join(music_folder, 'music3')

    # Create music1, music2, and music3 directories if they don't exist
    os.makedirs(music1_folder, exist_ok=True)
    os.makedirs(music2_folder, exist_ok=True)
    os.makedirs(music3_folder, exist_ok=True)

    # Get all mp3 files in the music folder
    mp3_files = [f for f in os.listdir(music_folder) if f.endswith('.mp3') and os.path.isfile(os.path.join(music_folder, f))]

    # Calculate the split points
    total_files = len(mp3_files)
    part_size = math.ceil(total_files / 3)

    music1_files = mp3_files[:part_size]
    music2_files = mp3_files[part_size:part_size * 2]
    music3_files = mp3_files[part_size * 2:]

    # Move files to music1
    for file in music1_files:
        shutil.move(os.path.join(music_folder, file), os.path.join(music1_folder, file))

    # Move files to music2
    for file in music2_files:
        shutil.move(os.path.join(music_folder, file), os.path.join(music2_folder, file))

    # Move files to music3
    for file in music3_files:
        shutil.move(os.path.join(music_folder, file), os.path.join(music3_folder, file))

    print(f"Moved {len(music1_files)} files to {music1_folder}")
    print(f"Moved {len(music2_files)} files to {music2_folder}")
    print(f"Moved {len(music3_files)} files to {music3_folder}")

if __name__ == '__main__':
    split_music_files()
