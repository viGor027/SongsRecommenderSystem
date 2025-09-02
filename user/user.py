from user.utils import SPACE_FILE_PATH, SPACE_INDEX_FILE_PATH
from user.space_search import find_k_closest
import torch
import random
import json

NUMBER_OF_RECOMMENDATIONS = 5

SPACE = torch.load(SPACE_FILE_PATH)
SPACE_INDEX = json.load(open(SPACE_INDEX_FILE_PATH))
TAGS = SPACE_INDEX["tags"]
TITLES = SPACE_INDEX["titles"]

TIME_STARTS = [0]

for i in range(1, len(TITLES)):
    if TITLES[i] == TITLES[i - 1]:
        TIME_STARTS.append(TIME_STARTS[-1] + 1)
    else:
        TIME_STARTS.append(0)


def print_fragment_data(idx: int, tags_only: bool = False) -> None:
    """
    Prints the data of the fragment with the given index.

    Args:
        idx (int): The index of the fragment.
        tags_only (bool): If True, only the tags of the fragment will be printed.
    """
    title = TITLES[idx]
    start_time = TIME_STARTS[idx]
    tags = TAGS[idx]

    print(f"{title}:")
    if not tags_only:
        print(f"\tStart time: {start_time}s | Idx: {idx} | Tags: {tags}")
    else:
        print(f"\tTags: {tags}")


class User:
    """
    A class representing a user of the system.

    Attributes:
        liked_songs (list): A list of indexes of songs liked by the user.
        liked_songs_space (list): A list of indexes of fragments of songs liked by the user.

    Methods:
        add_liked(index : int) -> None:
            Adds a song to the list of liked songs.
        add_liked_by_title(title : str) -> None:
            Adds a song to the list of liked songs by its title.
        get_recommendations() -> None:
            Prints the recommendations for the user.
    """

    def __init__(self) -> None:
        self.liked_songs = []
        self.liked_songs_space = []

    def add_liked(self, index: int) -> None:
        self.liked_songs.append(index)
        title = TITLES[index]
        for i, t in enumerate(TITLES):
            if t == title:
                self.liked_songs_space.append(i)

    def add_liked_by_title(self, title: str) -> None:
        for i in range(len(TITLES)):
            if TITLES[i] == title:
                self.add_liked(i)
                return

    def get_recommendations(self) -> None:
        liked_space = []

        if len(self.liked_songs) != 0:
            liked_space = random.sample(
                self.liked_songs_space,
                min(max(5, len(self.liked_songs_space) // 30), 30),
            )

        recommendations = find_k_closest(
            liked_space,
            SPACE,
            TITLES,
            NUMBER_OF_RECOMMENDATIONS,
            self.liked_songs_space,
        )

        liked_space = sorted(liked_space, key=lambda x: x)
        self.liked_songs = sorted(self.liked_songs, key=lambda x: x)

        print(50 * "-")
        print("Liked songs:")
        for l_song in self.liked_songs:
            print("Liked song ->", end=" ")
            print_fragment_data(l_song, tags_only=True)

        print("\nA random selection of fragments from songs liked by the user:")
        for i in liked_space:
            print("Selected fragment ->", end=" ")
            print_fragment_data(i)

        print("\nRecommendations:")
        for i, d in recommendations:
            print(f"Reccomended fragment with d={d}->", end=" ")
            print_fragment_data(i)


user = User()
while True:
    print("Enter command (l <index>, lt <title>, r):")
    line = input()
    line = line.split()
    if line[0] == "l":
        user.add_liked(int(line[1]))
    elif line[0] == "lt":
        user.add_liked_by_title(line[1])
    elif line[0] == "r":
        user.get_recommendations()
    else:
        break
