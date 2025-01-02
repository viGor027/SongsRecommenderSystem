from torch import nn
from model_components.temporal_compressor.conv1d_base_block import Conv1DBaseBlock


class Conv1DBlockWithDilationNoSkip(nn.Module):
    """
    A convolutional block that processes 1D inputs without skip connections, incorporating dilation.

    This class is implemented with causal padding(look at Conv1DBaseBlock implementation for further explanation).

    Notes:
        - Every instance of this block will compress the temporal dimension (length of the time axis) by a factor of 2.
    """

    def __init__(self, block_num: int, input_len: int,
                 n_input_channels: int, n_layers: int,
                 n_filters_per_layer: int, kernel_size: int,
                 stride: int):
        """
        Notes:
            - block_num indicates the sequential position of this block in the model.
            - input_len is a Length of the input's temporal dimension, corresponding to L_in in temporal_compressor/note.md.
            - n_input_channels is equal to n_mels if this is the first block in a model.
        """
        super().__init__()

        self.block = Conv1DBaseBlock(
            block_num=block_num, input_len=input_len,
            n_input_channels=n_input_channels, n_layers=n_layers,
            n_filters_per_layer=n_filters_per_layer, kernel_size=kernel_size,
            stride=stride, dilation=True
        )

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        return self.block(x)

    def debug_forward(self, x):
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            x = layer(x)
            print(f'Output shape {x.shape}')
            print()
        return x


if __name__ == "__main__":
    # Usage example
    import os
    import torch
    from song_pipeline.constants import N_SECONDS, N_MELS, PROJECT_FOLDER_DIR
    from song_pipeline.spectogram_pipeline import SpectogramPipeline

    sample_song_title = 'Retrospective_-_Alex_Skrindo__JJD.mp3'
    sample_song_path = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'music', sample_song_title)

    music_dir = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'music')
    songs_titles = os.listdir(music_dir)
    songs_paths = [os.path.join(music_dir, song) for song in songs_titles]

    ppl = SpectogramPipeline(music_dir)
    ppl.set_config(n_seconds=N_SECONDS, n_mels=N_MELS, spec_type='mel')

    song = ppl.get_song_specs(
        sample_song_path, sample_song_title,
        ['dummy_tag_1', 'dummy_tag_2'], return_dict=True
    )

    ith_sample = 4
    spec_sample = torch.Tensor(song['samples'][ith_sample:ith_sample + 1])
    print("Shape before: ", spec_sample.shape)
    model = Conv1DBlockWithDilationNoSkip(block_num=1, input_len=spec_sample.shape[-1],
                                          n_input_channels=N_MELS, kernel_size=2, stride=1,
                                          n_filters_per_layer=32, n_layers=3
                                          )
    # p = model(spec_sample)
    p = model.debug_forward(spec_sample)
    print("Shape after: ", p.shape)
    print("Resulting tensor: ")
    print(p)
