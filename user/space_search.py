from user.utils import SPACE_FOLDER_PATH, SPACE_FILE_PATH, SPACE_INDEX_FILE_PATH
import torch

def find_distances(index : int, space : torch.Tensor) -> torch.Tensor:
    """
    Function that finds distances between a given index and all other vectors in the data tensor.

    Parameters:
    index (int): Index of the vector in the data tensor.
    space (torch.Tensor): Data tensor.
    """
    dists = torch.cdist(space[index].unsqueeze(0), space)
    return dists.squeeze(0)

def find_k_closest(liked_songs : list, 
                   space : torch.Tensor, 
                   titles : list,
                   k : int,
                   excluded : list) -> torch.Tensor:
    """
    Function that finds k closest vectors to the given index in the data tensor.

    Parameters:
    liked_songs (list): List of indices of liked songs.
    space (torch.Tensor): Data tensor.
    titles (list): List of titles of the songs.
    k (int): Number of closest vectors to find.
    excluded (list): List of indices to exclude from the search.
    """
    dists_add = []
    for i in liked_songs:
        dists_add.append(find_distances(i, space))
    
    if len(dists_add) == 0:
        return []
    else:
        dists = torch.stack(dists_add).sum(0) 

    dists = dists.squeeze(0)
    dists[excluded] = float('inf')
    dists = [d.item() for d in dists]
    dists = [(i, titles[i], d) for i, d in enumerate(dists)]
    dists = sorted(dists, key=lambda x: x[2])
    
    recc_titles = []
    recc = []
    while k > 0 and len(dists) > 0:
        i, title, d = dists.pop(0)
        if title in recc_titles:
            continue
        if d == float('inf'):
            break
        recc_titles.append(title)
        recc.append((i, d))
        k -= 1

    return recc
