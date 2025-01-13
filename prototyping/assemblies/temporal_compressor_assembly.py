import torch
import torch.nn as nn
from model_components.classifier.base_classifier import BaseClassifier
from typing import Literal


class MainAssembly(nn.Module):
    """
    A wrapper for convenient assembling temporal compressors.

    Args:
       ConvCls (nn.Module): Convolutional block class used for feature extraction.
       n_blocks (int): Number of convolutional blocks.
       n_layers_per_block (list[int]): Number of layers per block.
       n_filters_per_block (list[int]): Number of filters in each block layer.
       n_filters_per_skip (list[int]): Number of skip connection filters per block.
       input_len (int): Length of the input sequence.
       n_input_channels (int): Number of input channels.
       n_classes (int): Number of output classes for the classifier.
       with_classifier (bool): Whether to pass input through classifier during forward.
   """
    def __init__(self, ConvCls, n_blocks: int,
                 n_layers_per_block: list[int],
                 n_filters_per_block: list[int],
                 n_filters_per_skip: list[int],
                 input_len: int, n_input_channels: int,
                 n_classes: int, reduction_strat: Literal['conv', 'max_pool', 'avg_pool'] = 'conv',
                 with_classifier: bool = True):
        super().__init__()

        self.ConvCls = ConvCls
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_filters_per_block = n_filters_per_block
        self.n_filters_per_skip = n_filters_per_skip
        self.reduction_strat = reduction_strat
        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes
        self.with_classifier = with_classifier

        self.conv = self.build_conv()
        self.classifier = self.build_class()

        self.classifier_inp = None

        self.forward_func = self._forward_with_classifier if with_classifier else self._forward_no_classifier

    def build_conv(self):
        """
        Builds temporal compressor based on model configuration.

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
                             n_layers=self.n_layers_per_block[i+1],
                             n_filters_per_layer=self.n_filters_per_block[i+1],
                             n_filters_skip=self.n_filters_per_skip[i+1],
                             reduction_strat=self.reduction_strat,
                             kernel_size=2, stride=1)
            )
            inp_len = inp_len // 2
        self.classifier_inp = inp_len * (
                self.n_filters_per_block[self.n_blocks-1] + self.n_filters_per_skip[self.n_blocks-1]
        )

        return nn.Sequential(*blocks)

    def build_class(self):
        """
        Builds the classifier for final prediction.

        Returns:
            BaseClassifier: A fully connected classifier.
        """
        n_layers = 5
        n_input_features = self.classifier_inp
        units_per_layer = [256, 224, 192, 160, 128]

        classifier = BaseClassifier(
            n_layers=n_layers,
            n_input_features=n_input_features,
            units_per_layer=units_per_layer,
            n_classes=self.n_classes
        )
        return classifier

    def forward(self, x):
        x = self.forward_func(x)
        return x

    def _forward_no_classifier(self, x):
        x = self.conv(x)
        return x

    def _forward_with_classifier(self, x):
        x = self.conv(x)
        x = x.reshape((x.size(0), -1))
        x = self.classifier(x)
        return x

    def save_instance(self):
        """Saves parameters of assembly"""
        return {
            'class_name': self.__class__.__name__,
            'parameters': {
                        "ConvCls": str(self.ConvCls),
                        "n_blocks": self.n_blocks,
                        "n_layers_per_block": self.n_layers_per_block,
                        "n_filters_per_block": self.n_filters_per_block,
                        "n_filters_per_skip": self.n_filters_per_skip,
                        "input_len": self.input_len,
                        "n_input_channels": self.n_input_channels,
                        "n_classes": self.n_classes,
                        "reduction_strat": self.reduction_strat,
                        "with_classifier": self.with_classifier
                        }
        }
