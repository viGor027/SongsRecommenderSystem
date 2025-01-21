import torch.nn as nn
from model_components.classifier.base_classifier import BaseClassifier
from typing import Literal


class CnnDenseAssembly(nn.Module):
    """
    A wrapper for convenient model assembling.

    You must initialize each part of the model before usage:
     '''
        model = CnnRnnDenseAssembly()
        model.init_conv(...)
        model.init_seq_encoder(...)
        model.init_classifier(...)
     '''
   """

    def __init__(self):
        """All the below attributes are set during initialization mentioned in class docstring."""
        super().__init__()

        self.ConvCls = None
        self.n_blocks = None
        self.n_layers_per_block = None
        self.n_filters_per_block = None
        self.n_filters_per_skip = None
        self.reduction_strat = None
        self.input_len = None
        self.n_input_channels = None

        self.seq_encoder_input_features = None
        self.n_seq_encoder_layers = None
        self.n_units_per_seq_encoder_layer = None
        self.n_embedding_dims = None

        self.n_classifier_layers = None
        self.n_units_per_classifier_layer = None
        self.n_classes = None

        self.conv = None
        self.seq_encoder = None
        self.classifier = None

    def init_conv(self, ConvCls, n_blocks: int,
                  n_layers_per_block: list[int],
                  n_filters_per_block: list[int],
                  n_filters_per_skip: list[int],
                  input_len: int, n_input_channels: int,
                  reduction_strat: Literal['conv', 'max_pool', 'avg_pool'] = 'conv'):
        """
        Args:
           ConvCls (nn.Module): Convolutional block class used for feature extraction.
           n_blocks (int): Number of convolutional blocks.
           n_layers_per_block (list[int]): Number of layers per block.
           n_filters_per_block (list[int]): Number of filters in each block layer.
           n_filters_per_skip (list[int]): Number of skip connection filters per block.
           input_len (int): Length of the input sequence.
           n_input_channels (int): Number of input channels.
           reduction_strat (Literal['conv', 'max_pool', 'avg_pool']): reduction strategy used by convolutional blocks
        """
        self.ConvCls = ConvCls
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_filters_per_block = n_filters_per_block
        self.n_filters_per_skip = n_filters_per_skip
        self.reduction_strat = reduction_strat
        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.conv = self._build_conv()

    def _build_conv(self):
        """
        Builds convolutional blocks based on configuration passed to init_conv.

        Returns:
            nn.Sequential: Sequential container of convolutional blocks.
        """
        blocks = [
            self.ConvCls(block_num=0,
                         input_len=self.input_len,
                         n_input_channels=self.n_input_channels,
                         n_layers=self.n_layers_per_block[0],
                         n_filters_per_layer=self.n_filters_per_block[0],
                         n_filters_skip=self.n_filters_per_skip[0],
                         reduction_strat=self.reduction_strat,
                         kernel_size=2, stride=1)
        ]
        inp_len = self.input_len // 2
        for i in range(self.n_blocks - 1):
            blocks.append(
                self.ConvCls(block_num=i + 1,
                             input_len=inp_len,
                             n_input_channels=self.n_filters_per_skip[i] + self.n_filters_per_block[i],
                             n_layers=self.n_layers_per_block[i + 1],
                             n_filters_per_layer=self.n_filters_per_block[i + 1],
                             n_filters_skip=self.n_filters_per_skip[i + 1],
                             reduction_strat=self.reduction_strat,
                             kernel_size=2, stride=1)
            )
            inp_len = inp_len // 2
        self.seq_encoder_input_features = inp_len * (
                self.n_filters_per_block[self.n_blocks - 1] + self.n_filters_per_skip[self.n_blocks - 1]
        )

        return nn.Sequential(*blocks)

    def init_seq_encoder(self, n_seq_encoder_layers: int,
                         n_units_per_seq_encoder_layer: list[int],
                         n_embedding_dims: int):
        """
        Initializes the sequence encoder with the specified parameters.

        Args:
            n_seq_encoder_layers (int): Number of layers in the sequence encoder.
            n_units_per_seq_encoder_layer (list[int]): Number of units per sequence encoder layer.
            n_embedding_dims (int): Dimension of the embeddings produced by the sequence encoder.
        """
        self.n_seq_encoder_layers = n_seq_encoder_layers
        self.n_units_per_seq_encoder_layer = n_units_per_seq_encoder_layer
        self.n_embedding_dims = n_embedding_dims
        self.seq_encoder = self._build_seq_encoder()

    def _build_seq_encoder(self):
        """
        Builds dense sequence encoder based on configuration passed to init_seq_encoder.

        Returns:
            BaseClassifier: Sequence encoder.
        """
        seq_encoder = BaseClassifier(
            n_layers=self.n_seq_encoder_layers,
            n_input_features=self.seq_encoder_input_features,
            units_per_layer=self.n_units_per_seq_encoder_layer,
            n_classes=self.n_embedding_dims,
            sigmoid_output=False
        )
        return seq_encoder

    def init_classifier(self, n_classifier_layers: int,
                        n_units_per_classifier_layer: list[int],
                        n_classes: int):
        """
        Args:
            n_classifier_layers (int): Number of layers in the classifier.
            n_units_per_classifier_layer (list[int]): List of the number of units per classifier layer.
            n_classes (int): Number of output classes for classification.
        """
        self.n_classifier_layers = n_classifier_layers
        self.n_units_per_classifier_layer = n_units_per_classifier_layer
        self.n_classes = n_classes
        self.classifier = self._build_class()

    def _build_class(self):
        """
        Builds sequence encoder based on configuration passed to init_classifier.

        Returns:
            BaseClassifier: Classifier network.
        """
        classifier = BaseClassifier(
            n_layers=self.n_classifier_layers,
            n_input_features=self.n_embedding_dims,
            units_per_layer=self.n_units_per_classifier_layer,
            n_classes=self.n_classes
        )
        return classifier

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((x.size(0), -1))
        x = self.seq_encoder(x)
        x = self.classifier(x)
        return x

    def get_instance_config(self):
        """
        Retrieves the configuration of the model instance.

        Returns:
            dict: A dictionary containing the model's configuration.
        """
        return {
            'class_name': self.__class__.__name__,
            'temporal_compressor': {
                "ConvCls": str(self.ConvCls).split('.')[-1][:-2],
                "input_len": self.input_len,
                "n_input_channels": self.n_input_channels,
                "n_blocks": self.n_blocks,
                "n_layers_per_block": self.n_layers_per_block,
                "n_filters_per_block": self.n_filters_per_block,
                "n_filters_per_skip": self.n_filters_per_skip,
                "reduction_strat": self.reduction_strat,
            },
            'sequence_encoder': {
                'n_seq_encoder_layers': self.n_seq_encoder_layers,
                'n_units_per_seq_encoder_layer': self.n_units_per_seq_encoder_layer,
                'n_embedding_dims': self.n_embedding_dims
            },
            'classifier': {
                "n_classifier_layers": self.n_classifier_layers,
                "n_units_per_classifier_layer": self.n_units_per_classifier_layer,
                "n_classes": self.n_classes
            }
        }
