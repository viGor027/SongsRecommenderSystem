import os
import shutil
import math
from song_pipeline.constants import SONGS_DIR


def split_music_files(num_folders):
    music_folder = SONGS_DIR

    folder_paths = []
    for i in range(1, num_folders + 1):
        folder_path = os.path.join(music_folder, f'music{i}')
        os.makedirs(folder_path, exist_ok=True)
        folder_paths.append(folder_path)

    mp3_files = [f for f in os.listdir(music_folder) if f.endswith('.mp3') and os.path.isfile(os.path.join(music_folder, f))]

    total_files = len(mp3_files)
    part_size = math.ceil(total_files / num_folders)

    for idx, folder_path in enumerate(folder_paths):
        start_idx = idx * part_size
        end_idx = start_idx + part_size
        files_to_move = mp3_files[start_idx:end_idx]

        for file in files_to_move:
            shutil.move(os.path.join(music_folder, file), os.path.join(folder_path, file))

        print(f"Moved {len(files_to_move)} files to {folder_path}")


if __name__ == '__main__':
    split_music_files(80)
