import torch
import torch.nn as nn
from model_components.temporal_compressor.conv1d_block_no_dilation_with_skip import Conv1DBlockNoDilationWithSkip
from model_components.classifier.base_classifier import BaseClassifier


class Assembly(nn.Module):
    def __init__(self, ConvCls, n_blocks: int, n_filters: int, n_skip: int, input_len: int, n_input_channels):
        super().__init__()

        self.ConvCls = ConvCls
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.n_skip = n_skip
        self.input_len = input_len
        self.n_input_channels = n_input_channels

        self.conv = self.build_conv()
        self.classifier = self.build_class()
        self.classifier_inp = None

    def build_conv(self):
        blocks = [
            self.ConvCls(block_num=0,
                         input_len=self.input_len,
                         n_input_channels=self.n_input_channels,
                         n_layers=2,
                         n_filters_per_layer=self.n_filters,
                         n_filters_skip=self.n_skip,
                         kernel_size=2, stride=1)
        ]
        inp_len = self.input_len // 2
        for i in range(self.n_blocks - 1):
            blocks.append(
                self.ConvCls(block_num=i + 1,
                             input_len=inp_len,
                             n_input_channels=self.n_filters + self.n_skip,
                             n_layers=2,
                             n_filters_per_layer=self.n_filters,
                             n_filters_skip=self.n_skip,
                             kernel_size=2, stride=1)
            )
            inp_len = inp_len // 2
        self.classifier_inp = inp_len

        return nn.Sequential(*blocks)

    def build_class(self):
        n_layers = 1
        n_input_features = self.classifier_inp * (self.n_filters + self.n_skip)
        units_per_layer = [4 * 32]
        n_classes = 91

        classifier = BaseClassifier(
            n_layers=n_layers,
            n_input_features=n_input_features,
            units_per_layer=units_per_layer,
            n_classes=n_classes
        )
        return classifier

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((x.size(0), -1))
        x = self.classifier(x)
        return x
