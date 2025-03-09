from enum import Enum

import cv2
import numpy as np
import torch
from cv2.dnn import NMSBoxes
from torch import nn
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class Dataset(Enum):
    COCO = 1

    def class_name_path(self):
        if self is self.COCO:
            return "model/coco.names"
        else:
            return None


class DarknetType(Enum):
    YOLO_V1 = 1
    YOLO_V2 = 2
    YOLO_V3 = 3

    def cfg_path(self):
        if self is self.YOLO_V1:
            return "model/yolov1.cfg"
        elif self is self.YOLO_V2:
            return "model/yolov2.cfg"
        elif self is self.YOLO_V3:
            return "model/yolov3.cfg"
        else:
            return None

    def weight_path(self):
        if self is self.YOLO_V1:
            return "model/yolov1.weights"
        elif self is self.YOLO_V2:
            return "model/yolov2.weights"
        elif self is self.YOLO_V3:
            return "model/yolov3.weights"
        else:
            return None


class Darknet:
    def __init__(self, model=DarknetType.YOLO_V3, data=Dataset.COCO):
        self.darknet = cv2.dnn.readNetFromDarknet(model.cfg_path(), model.weight_path())
        self.out_layers = self.darknet.getUnconnectedOutLayersNames()
        try:
            f = open(data.class_name_path(), 'r')
            self.class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise AssertionError(f"File not found: {data.class_name_path()}")

    def inspect_output_layers(self, img):
        blob_img = cv2.dnn.blobFromImage(img, 1. / 256, (448, 448), (0, 0, 0), swapRB=True)
        layerIds, inLayerShapes, outLayerShapes = self.darknet.getLayersShapes([blob_img.shape])
        out_layer_ids = self.darknet.getUnconnectedOutLayers()
        layer_names = ["__NetInputLayer__"] + list(self.darknet.getLayerNames())
        for layer_id in out_layer_ids:
            layer_name = layer_names[layer_id]
            input1_shape, input2_shape = inLayerShapes[layer_id]
            output_shape, = outLayerShapes[layer_id]
            print(f"{layer_name} : {input1_shape} + {input2_shape} -> {output_shape}")

    def forward(self, img):
        blob_img = cv2.dnn.blobFromImage(img, 1. / 256, (448, 448), (0, 0, 0), swapRB=True)
        self.darknet.setInput(blob_img)
        out_layers = self.darknet.getUnconnectedOutLayersNames()
        return self.darknet.forward(out_layers)
