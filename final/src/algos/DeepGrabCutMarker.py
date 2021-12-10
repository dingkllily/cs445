import os
import cv2
import sys
import torch
import numpy as np
from torch.nn.functional import upsample
from thirdparty.DeepGrabCut.dataloaders import utils
import thirdparty.DeepGrabCut.networks.deeplab_resnet as resnet

from copy import deepcopy


class DeepGrabCutMarker:
    def __init__(self, image=None):
        if image is None:
            self.image = np.ones([600, 800, 3], dtype=np.uint8) * 255

        else:
            self.image = image
        self.origsize = self.image.shape[:2]
        self.image = cv2.resize(self.image, (450, 450))
        self.segment = np.zeros_like(self.image, dtype=np.uint8)
        self.candidate = np.zeros_like(self.image, dtype=np.uint8)
        self.mask = np.zeros_like(self.image, dtype=np.uint8)

        self.start = [0, 0]
        self.end = [0, 0]
        self.current_seq = []
        self.labels = []
        self.line_thickness = 2 + np.min(self.image.shape[:2]) // 400

        self.gpu_id = 0
        self.device = torch.device("cuda:" + str(self.gpu_id) if torch.cuda.is_available() else "cpu")
        self.left = sys.maxsize
        self.right = 0
        self.up = sys.maxsize
        self.bottom = 0
        self.prev_x = -1
        self.prev_y = -1
        self.net = None
        self.thres = 0.8

    def reset_image(self, cvimage):
        self.image = cvimage
        self.origsize = self.image.shape[:2]
        self.image = cv2.resize(self.image, (450, 450))

        self.segment = np.zeros_like(self.image, dtype=np.uint8)
        self.candidate = np.zeros_like(self.image, dtype=np.uint8)
        self.mask = np.zeros_like(self.image, dtype=np.uint8)

        self.start = [0, 0]
        self.end = [0, 0]
        self.current_seq = []
        self.labels = []
        self.line_thickness = 2 + np.min(self.image.shape[:2]) // 400

    def load_model(self):
        self.net = resnet.resnet101(1, nInputChannels=4, classifier="psp")
        state_dict_checkpoint = torch.load(
            os.path.join("src/thirdparty/DeepGrabCut/models", "deepgc_pascal_epoch-99.pth"),
            map_location=lambda storage, loc: storage,
        )
        self.net.load_state_dict(state_dict_checkpoint)
        self.net.eval()
        self.net.to(self.device)

    def start_draw(self, x, y):
        self.prev_x, self.prev_y = x, y
        self.start_x, self.start_y = x, y
        self.left = min(sys.maxsize, x)
        self.right = max(0, x)
        self.up = min(sys.maxsize, y)
        self.bottom = max(0, y)

    def add_seed(self, x, y):
        prev = (self.prev_x, self.prev_y)
        cv2.line(self.candidate, prev, (x, y), (0, 0, 255), self.line_thickness)
        self.prev_x, self.prev_y = x, y
        self.left = min(self.left, x)
        self.right = max(self.right, x)
        self.up = min(self.up, y)
        self.bottom = max(self.bottom, y)

    def end_draw(self, x, y):
        prev = (self.prev_x, self.prev_y)
        cv2.line(self.candidate, prev, (x, y), (0, 0, 255), self.line_thickness)
        self.prev_x, self.prev_y = x, y
        prev = (self.prev_x, self.prev_y)
        start = (self.start_x, self.start_y)
        cv2.line(self.candidate, prev, start, (0, 0, 255), self.line_thickness)

    def get_seg_overlay(self):
        return cv2.addWeighted(self.image, 0.8, self.segment, 0.8, 0.1)

    def get_candidate_overlay(self):
        return cv2.addWeighted(self.image, 0.8, self.candidate, 0.8, 0.1)

    def calculate(self):
        if self.net is None:
            self.load_model()
        self.segment = np.zeros_like(self.image, dtype=np.uint8)

        mask_input = utils.fixed_resize(self.candidate, (450, 450)).astype(np.uint8)
        img_input = utils.fixed_resize(self.image, (450, 450)).astype(np.uint8)
        tmp = (mask_input[:, :, 2] > 0).astype(np.uint8)
        tmp_ = deepcopy(tmp)
        fill_mask = np.ones((tmp.shape[0] + 2, tmp.shape[1] + 2))
        fill_mask[1:-1, 1:-1] = tmp_
        fill_mask = fill_mask.astype(np.uint8)
        cv2.floodFill(tmp_, fill_mask, (int((self.left + self.right) / 2), int((self.up + self.bottom) / 2)), 5)
        tmp_ = tmp_.astype(np.int8)

        dismap = cv2.distanceTransform(
            tmp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )  # compute distance inside and outside bounding box
        dismap = tmp_ * dismap + 128

        dismap[dismap > 255] = 255
        dismap[dismap < 0] = 0
        dismap = dismap

        # dismap = dismap.astype(np.uint8)
        dismap = utils.fixed_resize(dismap, (450, 450)).astype(np.uint8)

        dismap = np.expand_dims(dismap, axis=-1)

        image = img_input[:, :, ::-1]  # change to rgb
        merge_input = np.concatenate((image, dismap), axis=2).astype(np.float32)
        inputs = torch.from_numpy(merge_input.transpose((2, 0, 1))[np.newaxis, ...])

        inputs = inputs.to(self.device)
        outputs = self.net.forward(inputs)
        outputs = upsample(outputs, size=(450, 450), mode="bilinear", align_corners=True)
        outputs = outputs.to(torch.device("cpu"))

        prediction = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        prediction = 1 / (1 + np.exp(-prediction))
        prediction = np.squeeze(prediction)
        prediction[prediction > self.thres] = 255
        prediction[prediction <= self.thres] = 0

        prediction = np.expand_dims(prediction, axis=-1).astype(np.uint8)
        image = image[:, :, ::-1]  # change to bgr
        display_mask = np.concatenate([np.zeros_like(prediction), np.zeros_like(prediction), prediction], axis=-1)
        self.segment = display_mask
        H, W = self.image.shape[:2]
        self.segment = cv2.resize(self.segment, (W, H))

    def clean_seeds(self):
        self.candidate = np.zeros_like(self.image, dtype=np.uint8)

    def get_mask(self):
        mask = cv2.resize(self.segment, (self.origsize[1], self.origsize[0]))[:, :, 2]

        return mask > 0
